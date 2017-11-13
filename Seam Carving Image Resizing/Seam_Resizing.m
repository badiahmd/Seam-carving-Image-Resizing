

function mainFunction
%choose the testing image
originalImg=imread('testImg1.jpg');
%originalImg=imread('testImg2.jpg');
%originalImg=imread('testImg3.jpg');
%originalImg=imread('testImg4.jpg');
%}

newRows = 350;
newColumns = 280;
originalImg=double(originalImg)/255;
[rows columns dimention]=size(originalImg);
resizeImg=originalImg;
%
%-----------------------Demonstration Purpose------------------------------


%find gradient image of luminance channel
gradientImg=sobelGradient(resizeImg);

    figure(2)
    imshow(gradientImg)

%finds seam calculation image
seamImg=seamMapping(gradientImg);
    figure(4)
    imshow(seamImg,[min(seamImg(:)) max(seamImg(:))])
    
% Finds seam    
SeamVector=vectorizedSeam(seamImg);

%plot image with superimposed seam
SeamedImg=superimposedSeam(gradientImg,SeamVector);
    figure(5)
    imshow(SeamedImg,[min(SeamedImg(:)) max(SeamedImg(:))])

%remove seam
SeamCutImg=removeSeam(resizeImg,SeamVector);

    figure(6)
    imshow(SeamCutImg)

%}

%to identify of how many vertical or horizontal seam needs to be 
%removed or added :
Rcols = columns-newColumns;
Rrows = rows-newRows;

if Rcols>0 % remove the vertical seams
    clear X
    %find a set of seam mapping from gradient, to be removed from the image
    X=seamRemovalMapping(resizeImg,Rcols);
    %remove the column that contains seam,then merge it .
    resizeImg=removeSeam(resizeImg,X);
    [rows columns dimention]=size(resizeImg);
elseif Rcols<0 %adding vertical seams
    clear X
    %find a set of seam mapping from gradient, to be added to the image
    X=seamRemovalMapping(resizeImg,abs(Rcols));
    %find the column that needs to be added with the seam,then merge it.
    resizeImg=mergeSeam(resizeImg,X);
    [rows columns dimention]=size(resizeImg);
end

if Rrows>0  %remove horizontal seams
    clear X
    %inverse rotated the image
    Y=permute(resizeImg,[2,1,3]);
    %find a set of seam mapping from inverse rotated image gradient,
    %to be removed from the image
    X=seamRemovalMapping(Y,Rrows);
    %find the column that needs to be removed with the seam,
    %then merge it and reverse the rotation.
    resizeImg=permute(removeSeam(Y,X),[2,1,3]);
    [rows columns dimention]=size(resizeImg);
elseif Rrows<0  %adding horizontal seams
    clear X
    %inverse rotated the image
    Y=permute(resizeImg,[2,1,3]);
    %find a set of seam mapping from inverse rotated image gradient,
    %to be removed from the image
    X=seamRemovalMapping(Y,abs(Rrows));
    %find the column that needs to be added with the seam,
    %then merge it and reverse the rotation.
    resizeImg=permute(mergeSeam(Y,X),[2,1,3]);
    seamImg=seamMapping(resizeImg);
    [rows columns dimention]=size(resizeImg);
end

figure,imshow(resizeImg);
figure,imshow(originalImg);


%------------------------Functions Implementation--------------------------

%find the energy-map using sobel operator
function imageGradient=sobelGradient(image)
%get the rows,colums,and dimention of the image
[rows columns dimention]=size(image);
%sobel operator is selected to calculate gradient image
Grd=[ -1 -2 -1;
       0  0  0;
       1  2  1];

%create 0 vectors with image vectors size(rows and colums)
imageGradient=zeros(rows,columns);
for i=1:dimention
    %compute central part of convolution of the same size image
    %with the sobel operator(Horizontal)
    horizontalGrad(:,:,i)=conv2(image(:,:,i),Grd,'same');
    %comput central part of convolution of the same size image
    %with the sobel operator(Vertically/transposed)
    verticalGrad(:,:,i)=conv2(image(:,:,i),Grd.','same');
    %find the sum of absolute horizontal and vertical gradient
    gradients(:,:,i)=abs(horizontalGrad(:,:,i))+abs(verticalGrad(:,:,i));
end
imageGradient=1/dimention*sum(gradients,3); %find average gradient magnitude image for RGB image

%find the seam mapping of the gradient image
function seamMap=seamMapping(gradientImage)

%get the rows,colums,and dimention of the gradient image
[rows columns]=size(gradientImage);

%create 0 vectors with image vectors size(rows and colums)
seamMap=zeros(rows,columns);
seamMap(1,:)=gradientImage(1,:);

%in order to finds te seam map, indexing are integrated as below, 
%given that the i are rows and j are the columns :
%[(i-1,j-1) (i-1,j) (i-1,j+1)]
%[(i,j-1) (i,j) (i,j+1)]
%[(i+1,j-1) (i+1,j) (i+1,j+1)]
for i=2:rows
    for j=1:columns
        if j-1<1
            seamMap(i,j)= gradientImage(i,j)+min([seamMap(i-1,j),seamMap(i-1,j+1)]);
        elseif j+1>columns

            seamMap(i,j)= gradientImage(i,j)+min([seamMap(i-1,j-1),seamMap(i-1,j)]);
        else

            seamMap(i,j)= gradientImage(i,j)+min([seamMap(i-1,j-1),seamMap(i-1,j),seamMap(i-1,j+1)]);
        end
    end
end

%returns vectors for the pixel to be removed from mapped seam images
function SeamVector=vectorizedSeam(mappedSeamImg);

%each 'j' of seamVector represents column value to be removed from row 'i'
[rows columns]=size(mappedSeamImg);
for i=rows:-1:1
    if i==rows
         %find the minimum value of last row
        [value, j]=min(mappedSeamImg(rows,:)); 
    else    %to overcome boundary issues
        if SeamVector(i+1)==1
            Vector=[Inf mappedSeamImg(i,SeamVector(i+1)) ...
                    mappedSeamImg(i,SeamVector(i+1)+1)];
        elseif SeamVector(i+1)==columns
            Vector=[mappedSeamImg(i,SeamVector(i+1)-1) ...
                    mappedSeamImg(i,SeamVector(i+1)) Inf];
        else
            Vector=[mappedSeamImg(i,SeamVector(i+1)-1) ...
                    mappedSeamImg(i,SeamVector(i+1)) ...
                    mappedSeamImg(i,SeamVector(i+1)+1)];
        end
        %find minimum index and value of 3 neighboring pixels in previous rows.
        [Value Index]=min(Vector);
        IndexIncrement=Index-2;
        j=SeamVector(i+1)+IndexIncrement;
    end
    SeamVector(i,1)=j;
end

%to find ordered set of vertical seams map,
%that are removed from image and return them in an array,
%given the columns or rows number in array suits to the seam to be removed
function seamVectors=seamRemovalMapping(image,vectors);

[rows columns dimention]=size(image);

E=sobelGradient(image);    %Finds the gradient image

for i=1:min(vectors,columns-1)

    %find "energy map" image used for seam calculation given the gradient image
    S=seamMapping(E);

    %find seam vector given input "energy map" seam calculation image
    seamVectors(:,i)=vectorizedSeam(S);

    %remove seam from image
    image=removeSeam(image,seamVectors(:,i));
    E=removeSeam(E,seamVectors(:,i));

    %updates size of image
    [rows columns dimention]=size(image);
end

%take input image and seamVector vectors to remove the seams by
%finding the pixels that contained the seams
function image=removeSeam(image,seamVector)

% Each col of seamVector must be a single seam.
[rows columns dimention]=size(image);
[sVectorRows sVectorColumns sVectorDimentions]=size(seamVector);

%to check the image and seamVector,if there is a dimension mismatch
if rows~=sVectorRows
    error('image and seamVector rows mismatch');
end

for k=1:sVectorColumns %loops through the set of seams
    for i=1:dimention  %if the image is RGB,go through each channel
        for j=1:rows %go through each row in image
            if seamVector(j,k)==1
                removeImg(j,:,i)=[image(j,2:columns,i)];
            elseif seamVector(j,k)==columns
                removeImg(j,:,i)=[image(j,1:columns-1,i)];
            else
                removeImg(j,:,i)=[image(j,1:seamVector(j,k)-1,i)...
                    image(j,seamVector(j,k)+1:columns,i)];
            end
        end
    end
    image=removeImg;
    clear removeImg
    [rows columns dimention]=size(image);
end


%for display purpose,this function to see the superimposed line of the seam
function image=superimposedSeam(image,seamVector)
% superimposedSeam takes as input an image and the seamVector array and produces
% an image with the seam line superimposed upon the input image, image, for
% display purposes.

values=1.5*max(image(:));
for i=1:size(seamVector,1)
    image(i,seamVector(i))=values;
end

%takes an input image and seamVector to add seams of interpolated pixels by
%finding the pixels of the seam
function image=mergeSeam(image,SeamVector)

[rows columns dimention]=size(image);
[sVectorRows sVectorColomns sVectorDimention]=size(SeamVector);

if rows~=sVectorRows
    error('SeamVector and image dimension mismatch');
end

for k=1:sVectorColomns %loops through the set of seams
    for i=1:dimention %if the image is RGB,go through each channel
        for j=1:rows %go through each row in image
            if SeamVector(j,k)==1
                mergeImg(j,:,i)=[image(j,1,i) image(j,1:columns,i)];
            elseif SeamVector(j,k)==columns
                mergeImg(j,:,i)=[image(j,1:columns,i) image(j,columns,i)];
            else
                mergeImg(j,:,i)=[image(j,1:SeamVector(j,k),i)...
                    1/2*(image(j,SeamVector(j,k),i)...
                    +image(j,SeamVector(j,k)+1,i))...
                    image(j,SeamVector(j,k)+1:columns,i)];
            end
        end
    end
    image=mergeImg;
    clear mergeImg
    [rows columns dimention]=size(image);
end