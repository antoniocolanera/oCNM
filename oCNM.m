classdef (ConstructOnLoad) oCNM < dynamicprops
    % oCNM  Orbital Cluster-based Network Modelling with STFT embedding
    %
    %   This class implements Orbital CNM with  Short-Time Fourier Transform
    %   (STFT) basis. The real and imaginary parts of the
    %   STFT are used as input features for the CNM model.
    %
    %   The oCNM model can be used to reconstruct multi-dimensional,
    %   non-stationary time signals by capturing their orbital evolution
    %   in the frequency domain.
    %
    %   USAGE:
    %     model = oCNM(X, t, nseg, novl, K, L, interp_method)
    %
    %   INPUTS:
    %     X             : (Nt x nx) input signal matrix (time x variables)
    %     t             : (Nt x 1) time vector
    %     nseg          : STFT segment length
    %     novl          : STFT overlap length
    %     K             : number of clusters
    %     L             : Markov model order
    %     interp_method : (string) interpolation method for CNM ('spline', 'linear', etc.)
    %
    %   METHODS:
    %     predict() : Predicts and reconstructs the original signal from STFT trajectory
    %
    %   EXAMPLE:
    %     model = oCNM(X, t, 128, 96, 10, 2, 'spline');
    %     Xrec = model.predict();
    %   REFERENCES:
    %     A. Colanera, N. Deng, M. Chiatto, L. de Luca, B. R. Noack.
    %     "Orbital cluster-based network modelling", COMPHY, 2025.
    %
    %   -----------------------------------------------------------------------
    %   Author: Antonio Colanera
    %   Created: July 2024
    %   -----------------------------------------------------------------------
    properties
        fs              % Sampling frequency
        win             % STFT window
        novl            % STFT overlap
        nseg            % Segment length
        nf              % Number of frequency bins
        nt2             % Number of STFT time steps
        stft_shape      % Shape of STFT matrix [nf, nt2, nx]
        Xstft_original  % STFT of original signal
        Xstft_ocnm
        fvec
        tvec
        dt2
        t
        nx
        TIME
        reduced_state   %  (real + imag STFT)
        cnm
    end

    methods
        function obj = oCNM(X, t, nseg, novl, K, L, interp_method)
            % Preparazione preliminare PRIMA di chiamare il costruttore CNM
            dt = t(2) - t(1);
            fs = 1 / dt;
            win = hamming(nseg, 'periodic');

            % Calcolo STFT
            [Xstft, fvec, tvec] = stft(X', fs, ...
                'Window', win, 'OverlapLength', novl, 'FrequencyRange', 'onesided');
            tvec = tvec - tvec(1);
            [nf,nt2,nx]=size(Xstft);

            reduced_stateMYc =[Xstft(:,:,1);Xstft(:,:,2)];



            reduced_stateMYRe= real(reduced_stateMYc);
            reduced_stateMYImm= imag(reduced_stateMYc);
            reduced_state =[reduced_stateMYRe;reduced_stateMYImm];

            dt2 = tvec(10) - tvec(9);

            %
            rng(1)
            obj.cnm=CNM(reduced_state, dt2, K, L, interp_method);
            obj.reduced_state = reduced_state;

            % 
            obj.dt2=dt2;
            obj.fs = fs;
            obj.nseg = nseg;
            obj.novl = novl;
            obj.nx = nx;
            obj.win = win;
            obj.Xstft_original = Xstft;
            obj.fvec = fvec;
            obj.tvec = tvec;
            obj.nt2 = nt2;
            obj.nf = nf;
            obj.stft_shape = size(Xstft);
            obj.t = tvec; % 
        end

        function Xrec = predict(obj)
            % Predict in STFT space and reconstruct time-domain signal
            initial_state = obj.reduced_state(:,1);
            prediction = obj.cnm.predict(initial_state, (obj.nt2-1)*obj.dt2, obj.dt2);

            nf = obj.nf;
            nx = obj.nx;
            nt2 = obj.nt2;

            XrecSTFT = zeros(nf, nt2, nx);
            XrecSTFTm=prediction(1:nx*nf,1:nt2) + prediction(nx*nf+1:end,1:nt2) *1j;

            XrecSTFT(:,:,1)=XrecSTFTm(1:nf,:);
            XrecSTFT(:,:,2)=XrecSTFTm(nf+1:2*nf,:);



            [Xrec,TIME] = istft(XrecSTFT, obj.fs, 'Window', obj.win, 'OverlapLength', obj.novl, 'FrequencyRange', 'onesided');
            Xrec = Xrec';
            obj.Xstft_ocnm=XrecSTFT;
            obj.TIME=TIME;
        end
    end
end
