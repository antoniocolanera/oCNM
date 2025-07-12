function R = autocorrelation(t,x,time_blocks,method)
n_blocks = floor(length(t) / time_blocks);
if n_blocks == 0
n_blocks = 1;
x_split = mat2cell(x,size(x,1));
else
x_split = mat2cell(x,[n_blocks*ones(1,fix((size(x,1)-1)/n_blocks)),1+rem(size(x,1)-1,n_blocks)]);
max_size = max(cellfun(@(elt) size(elt, 1), x_split));
min_size = min(cellfun(@(elt) size(elt, 1), x_split));
if max_size == min_size
else
x_split = x_split(1:end-1,:);
end
end
% Loop over the blocks
for i_block = 1:size(x_split,2)
x_block = x_split{:,i_block};
x_block = x_block - mean(x_block);
if strcmp(method, 'fft')
r = autocorrelation_fft(x_block);
elseif strcmp(method, 'dot')
r = autocorrelation_dot(x_block);
end
if i_block == 1
R = r;
else
R = R + r;
end
end
R = R / numel(x_split);
end


function R = autocorrelation_dot(x)
n = size(x, 1);
corr_matrix = x * x.';
r = zeros(n, 1);
for i = 1:n
r(i) = sum(diag(corr_matrix, i)) / (n-i);
end
R = r;
end
function r = autocorrelation_fft(x)
n = size(x, 1);
ext_size = 2*n - 1;
fsize = 2^ceil(log2(ext_size));
for i_dim = 1:size(x, 2)
cf = fft(x(:, i_dim), fsize);
sf = conj(cf) .* cf;
corr = ifft(sf, 'symmetric');
corr = corr / n;
if i_dim == 1
r = corr(1:n);
else
r = r + corr(1:n);
end
end
end