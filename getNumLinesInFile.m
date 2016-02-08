%
% return number of lines in file specified by filename
%
function n = getNumLinesInFile(filename)
fid = fopen(filename);
n = 0;
tline = fgetl(fid);
while ischar(tline)
    tline = fgetl(fid);
    n = n+1;
end
fclose(fid);
