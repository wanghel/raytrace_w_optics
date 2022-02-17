function average = AverageSpectrumSamples(lambda, vals, n, lambdastart, lambdaend)
sum = 0;
if lambdastart < lambda(1) 
    sum = sum + vals(1) * (lambda(1) - lambdastart);
end
if lambdaend > lambda(n - 1)
    sum = sum + vals(n - 1) * (lambdaend - lambda(n - 1));
end

i = 1;
while lambdastart > lambda(i+1) && i+1<n
    i = i+1;
end
while i + 1 < n && lambdaend >= lambda(i)
    seglambdastart = max(lambdastart, lambda(i));
    seglambdaend = min(lambdaend, lambda(i+1));
    sum = sum + 0.5 * (interp(lambda, seglambdastart, i, vals) + ...
        interp(lambda, seglambdaend, i, vals)) * (seglambdaend-seglambdastart);
    i = i + 1;
end
average = sum / (lambdaend - lambdastart);
end

function result =  interp(lambda, w, i, vals)
    result = lerp((w-lambda(i))/(lambda(i+1)-lambda(i)), vals(i), vals(i+1));
end