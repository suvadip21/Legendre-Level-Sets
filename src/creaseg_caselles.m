% Copyright or © or Copr. CREATIS laboratory, Lyon, France.
% 
% Contributor: Olivier Bernard, Associate Professor at the french 
% engineering university INSA (Institut National des Sciences Appliquees) 
% and a member of the CREATIS-LRMN laboratory (CNRS 5220, INSERM U630, 
% INSA, Claude Bernard Lyon 1 University) in France (Lyon).
% 
% Date of creation: 8th of October 2009
% 
% E-mail of the author: olivier.bernard@creatis.insa-lyon.fr
% 
% This software is a computer program whose purpose is to evaluate the 
% performance of different level-set based segmentation algorithms in the 
% context of image processing (and more particularly on biomedical 
% images).
% 
% The software has been designed for two main purposes. 
% - firstly, CREASEG allows you to use six different level-set methods. 
% These methods have been chosen in order to work with a wide range of 
% level-sets. You can select for instance classical methods such as 
% Caselles or Chan & Vese level-set, or more recent approaches such as the 
% one developped by Lankton or Bernard.
% - finally, the software allows you to compare the performance of the six 
% level-set methods on different images. The performance can be evaluated 
% either visually, or from measurements (either using the Dice coefficient 
% or the PSNR value) between a reference and the results of the 
% segmentation.
%  
% The level-set segmentation platform is citationware. If you are 
% publishing any work, where this program has been used, or which used one 
% of the proposed level-set algorithms, please remember that it was 
% obtained free of charge. You must reference the papers shown below and 
% the name of the CREASEG software must be mentioned in the publication.
% 
% CREASEG software
% "T. Dietenbeck, M. Alessandrini, D. Friboulet, O. Bernard. CREASEG: a
% free software for the evaluation of image segmentation algorithms based 
% on level-set. In IEEE International Conference On Image Processing. 
% Hong Kong, China, 2010."
%
% Bernard method
% "O. Bernard, D. Friboulet, P. Thevenaz, M. Unser. Variational B-Spline 
% Level-Set: A Linear Filtering Approach for Fast Deformable Model 
% Evolution. In IEEE Transactions on Image Processing. volume 18, no. 06, 
% pp. 1179-1191, 2009."
% 
% Caselles method
% "V. Caselles, R. Kimmel, and G. Sapiro. Geodesic active contours. 
% International Journal of Computer Vision, volume 22, pp. 61-79, 1997."
% 
% Chan & Vese method
% "T. Chan and L. Vese. Active contours without edges. IEEE Transactions on
% Image Processing. volume10, pp. 266-277, February 2001."
% 
% Lankton method
% "S. Lankton, A. Tannenbaum. Localizing Region-Based Active Contours. In 
% IEEE Transactions on Image Processing. volume 17, no. 11, pp. 2029-2039, 
% 2008."
% 
% Li method
% "C. Li, C.Y. Kao, J.C. Gore, Z. Ding. Minimization of Region-Scalable 
% Fitting Energy for Image Segmentation. In IEEE Transactions on Image 
% Processing. volume 17, no. 10, pp. 1940-1949, 2008."
% 
% Shi method
% "Yonggang Shi, William Clem Karl. A Real-Time Algorithm for the 
% Approximation of Level-Set-Based Curve Evolution. In IEEE Transactions 
% on Image Processing. volume 17, no. 05, pp. 645-656, 2008."
% 
% This software is governed by the BSD license and
% abiding by the rules of distribution of free software.
% 
% As a counterpart to the access to the source code and rights to copy,
% modify and redistribute granted by the license, users are provided only
% with a limited warranty and the software's author, the holder of the
% economic rights, and the successive licensors have only limited
% liability. 
% 
% In this respect, the user's attention is drawn to the risks associated
% with loading, using, modifying and/or developing or reproducing the
% software by the user in light of its specific status of free software,
% that may mean that it is complicated to manipulate, and that also
% therefore means that it is reserved for developers and experienced
% professionals having in-depth computer knowledge. Users are therefore
% encouraged to load and test the software's suitability as regards their
% requirements in conditions enabling the security of their systems and/or 
% data to be ensured and, more generally, to use and operate it in the 
% same conditions as regards security.
% 
%------------------------------------------------------------------------

%------------------------------------------------------------------------
% Description: This code implements the paper: "Geodesic active contours. 
% International Journal of Computer Vision." By Vincent Caselles.
%
% Coded by: Olivier Bernard (www.creatis.insa-lyon.fr/~bernard)
%------------------------------------------------------------------------


function [seg,phi,its] = creaseg_caselles(img,init_mask,max_its,propag,thresh,color,display)
  
    %-- default value for parameter img and init_mask
    if(~exist('img','var')) 
        img = imread('data/Image/simu1.bmp');   
        init_mask = repmat(0,[size(img,1) size(img,2)]);
        init_mask(round(size(img,1)/3):size(img,1)-round(size(img,1)/3),...
            round(size(img,2)/3):size(img,2)-round(size(img,2)/3)) = 1;
    end 
    %-- default value for parameter max_its is 100
    if(~exist('max_its','var')) 
        max_its = 100; 
    end 
    %-- default value for parameter propag is 1
    if(~exist('propag','var')) 
        propag = 1; 
    end 
    %-- default value for parameter max_its is 1
    if(~exist('thresh','var')) 
        thresh = 0; 
    end
    %-- default value for parameter color is 'r'
    if(~exist('color','var')) 
        color = 'r'; 
    end      
    %-- default behavior is to display intermediate outputs
    if(~exist('display','var'))
        display = true;
    end


    %-- ensures image is 2D double matrix
    img = im2graydouble(img);
        
%     init_mask = init_mask<=0;
    
    %-- Create a signed distance map (SDF) from mask
    phi = mask2phi(init_mask);

    
    %-- Compute feature image from gradient information
    h = fspecial('gaussian',[5 5],1);
    feature = imfilter(img,h,'same');
    [FX,FY] = gradient(feature);
    feature = sqrt(FX.^2+FY.^2+eps);
    feature = 1 ./ ( 1 + feature.^2 );
    
    
    %--
    fig = findobj(0,'tag','creaseg');
    ud = get(fig,'userdata');    
    
    
    %--main loop
    its = 0;      stop = 0;
    prev_mask = init_mask;        c = 0;

    while ((its < max_its) && ~stop)

        idx = find(phi <= 1.2 & phi >= -1.2);  %-- get the curve's narrow band
        
        if ~isempty(idx)
            %-- intermediate output
            if (display>0)
                if ( mod(its,50)==0 )
                    set(ud.txtInfo1,'string',sprintf('iteration: %d',its),'color',[1 1 0]);
                    showCurveAndPhi(phi,ud,color);
                    drawnow;
                end
            else
                if ( mod(its,10)==0 )            
                    set(ud.txtInfo1,'string',sprintf('iteration: %d',its),'color',[1 1 0]);
                    drawnow;
                end
            end
            
            %-- force from image information
            F = feature(idx);
            [curvature,normGrad,FdotGrad] = ...
                get_evolution_functions(phi,feature,idx);  % force from curvature penalty

            %-- gradient descent to minimize energy
            dphidt1 = F.*curvature.*normGrad;
            dphidt1 = dphidt1./max(abs(dphidt1(:)));
            
            dphidt2 = FdotGrad;
            dphidt2 = dphidt2./max(abs(dphidt2(:)));
            
            dphidt3 = F.*normGrad;
            dphidt3 = dphidt3./max(abs(dphidt3(:)));
            
            dphidt = dphidt1 + dphidt2 - propag*dphidt3;

            %-- maintain the CFL condition
            dt = .45/(max(abs(dphidt))+eps);

            %-- evolve the curve
            phi(idx) = phi(idx) + dt.*dphidt;

            %-- Keep SDF smooth
            phi = sussman(phi, .5);

            new_mask = phi<=0;
            c = convergence(prev_mask,new_mask,thresh,c);
            if c <= 5
                its = its + 1;
                prev_mask = new_mask;
            else stop = 1;
            end      
        else
            break;
        end
        
    end

    %-- final output
    showCurveAndPhi(phi,ud,color);

    %-- make mask from SDF
    seg = phi<=0; %-- Get mask from levelset

  
%---------------------------------------------------------------------
%---------------------------------------------------------------------
%-- AUXILIARY FUNCTIONS ----------------------------------------------
%---------------------------------------------------------------------
%---------------------------------------------------------------------
  
  
%-- Displays the image with curve superimposed
function showCurveAndPhi(phi,ud,cl)

	axes(get(ud.imageId,'parent'));
	delete(findobj(get(ud.imageId,'parent'),'type','line'));
	hold on; [c,h] = contour(phi,[0 0],cl{1},'Linewidth',3); hold off;
	delete(h);
    test = isequal(size(c,2),0);
	while (test==false)
        s = c(2,1);
        if ( s == (size(c,2)-1) )
            t = c;
            hold on; plot(t(1,2:end)',t(2,2:end)',cl{1},'Linewidth',3);
            test = true;
        else
            t = c(:,2:s+1);
            hold on; plot(t(1,1:end)',t(2,1:end)',cl{1},'Linewidth',3);
            c = c(:,s+2:end);
        end
	end    
    
  
%-- converts a mask to a SDF
function phi = mask2phi(init_a)

    phi=bwdist(init_a)-bwdist(1-init_a)+im2double(init_a)-.5;
  
%-- compute curvature along SDF
function [curvature,normGrad,FdotGrad] = get_evolution_functions(phi,feature,idx)

    [dimy, dimx] = size(phi);        
    [y x] = ind2sub([dimy,dimx],idx);  % get subscripts

    %-- get subscripts of neighbors
    ym1 = y-1; xm1 = x-1; yp1 = y+1; xp1 = x+1;

    %-- bounds checking  
    ym1(ym1<1) = 1; xm1(xm1<1) = 1;              
    yp1(yp1>dimy)=dimy; xp1(xp1>dimx) = dimx;    

    %-- get indexes for 8 neighbors
    idup = sub2ind(size(phi),yp1,x);    
    iddn = sub2ind(size(phi),ym1,x);
    idlt = sub2ind(size(phi),y,xm1);
    idrt = sub2ind(size(phi),y,xp1);
    idul = sub2ind(size(phi),yp1,xm1);
    idur = sub2ind(size(phi),yp1,xp1);
    iddl = sub2ind(size(phi),ym1,xm1);
    iddr = sub2ind(size(phi),ym1,xp1);
    
    %-- get central derivatives of SDF at x,y
    phi_x  = (-phi(idlt)+phi(idrt))/2;
    phi_y  = (-phi(iddn)+phi(idup))/2;
    phi_xx = phi(idlt)-2*phi(idx)+phi(idrt);
    phi_yy = phi(iddn)-2*phi(idx)+phi(idup);
    phi_xy = 0.25*phi(iddl)+0.25*phi(idur)...
             -0.25*phi(iddr)-0.25*phi(idul);
    phi_x2 = phi_x.^2;
    phi_y2 = phi_y.^2;
    
    %-- compute curvature (Kappa)
    curvature = ((phi_x2.*phi_yy + phi_y2.*phi_xx - 2*phi_x.*phi_y.*phi_xy)./...
              (phi_x2 + phi_y2 +eps).^(3/2));        

    %-- compute norm of gradient
    phi_xm = phi(idx)-phi(idlt);
    phi_xp = phi(idrt)-phi(idx);
    phi_ym = phi(idx)-phi(iddn);
    phi_yp = phi(idup)-phi(idx);    
    normGrad = sqrt( (max(phi_xm,0)).^2 + (min(phi_xp,0)).^2 + ...
        (max(phi_ym,0)).^2 + (min(phi_yp,0)).^2 );
    
    %-- compute scalar product between the feature image and the gradient of phi
    F_x = 0.5*feature(idrt)-0.5*feature(idlt);
    F_y = 0.5*feature(idup)-0.5*feature(iddn);   
    FdotGrad = (max(F_x,0)).*(phi_xp) + (min(F_x,0)).*(phi_xm) + ...
        (max(F_y,0)).*(phi_yp) + (min(F_y,0)).*(phi_ym);    
    
          
%-- Converts image to one channel (grayscale) double
function img = im2graydouble(img)

    [dimy, dimx, c] = size(img);
    if (isfloat(img))
        if (c==3) 
            img = rgb2gray(uint8(img)); 
        end
    else
        if (c==3) 
            img = rgb2gray(img); 
        end
        img = double(img);
    end

%-- level set re-initialization by the sussman method
function D = sussman(D, dt)
    
    % forward/backward differences
    a = D - shiftR(D); % backward
    b = shiftL(D) - D; % forward
    c = D - shiftD(D); % backward
    d = shiftU(D) - D; % forward

    a_p = a;  a_n = a; % a+ and a-
    b_p = b;  b_n = b;
    c_p = c;  c_n = c;
    d_p = d;  d_n = d;

    a_p(a < 0) = 0;
    a_n(a > 0) = 0;
    b_p(b < 0) = 0;
    b_n(b > 0) = 0;
    c_p(c < 0) = 0;
    c_n(c > 0) = 0;
    d_p(d < 0) = 0;
    d_n(d > 0) = 0;

    dD = zeros(size(D));
    D_neg_ind = find(D < 0);
    D_pos_ind = find(D > 0);
    dD(D_pos_ind) = sqrt(max(a_p(D_pos_ind).^2, b_n(D_pos_ind).^2) ...
                       + max(c_p(D_pos_ind).^2, d_n(D_pos_ind).^2)) - 1;
    dD(D_neg_ind) = sqrt(max(a_n(D_neg_ind).^2, b_p(D_neg_ind).^2) ...
                       + max(c_n(D_neg_ind).^2, d_p(D_neg_ind).^2)) - 1;

    D = D - dt .* sussman_sign(D) .* dD;
  
%-- whole matrix derivatives
function shift = shiftD(M)
    shift = shiftR(M')';

function shift = shiftL(M)
    shift = [ M(:,2:size(M,2)) M(:,size(M,2)) ];

function shift = shiftR(M)
  shift = [ M(:,1) M(:,1:size(M,2)-1) ];

function shift = shiftU(M)
    shift = shiftL(M')';
  
function S = sussman_sign(D)
    S = D ./ sqrt(D.^2 + 1);    


% Convergence Test
function c = convergence(p_mask,n_mask,thresh,c)
    diff = p_mask - n_mask;
    n_diff = sum(abs(diff(:)));
    if n_diff < thresh
        c = c + 1;
    else c = 0;
    end
    
