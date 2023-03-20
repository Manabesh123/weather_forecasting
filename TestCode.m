load('TEST_MATRIX.mat');

CTEMP = (TEMPCHENNAI-32)*5/9;

TEMPCHENNAI = round(CTEMP*10)/10;

Xi = TIME;
Yi = interp1(TIME, TEMPCHENNAI, Xi, 'spline');

TEMPCHENNAI = Yi;

DATA (20,366) = 0;
k = 1;
for i = 1: 20
    for j = 1 : 366
        DATA (i , j) = TEMPCHENNAI (k);
        k = k + 1;
    end
end

ip = DATA(1:19, :);
op = DATA(20, :);

NNDATA (5, 5856) = 0;
NNDATA (1, 1:5856) = TEMPCHENNAI (1: 5856);
NNDATA (2, 1:5856) = TEMPCHENNAI (367: 6222);
NNDATA (3, 1:5856) = TEMPCHENNAI (733: 6588);
NNDATA (4, 1:5856) = TEMPCHENNAI (1099: 6954);
NNDATA (5, 1:5856) = TEMPCHENNAI (1465: 7320);

input = NNDATA(1:4, :);
output = NNDATA(5, :);

clear Xi Yi i j k;
