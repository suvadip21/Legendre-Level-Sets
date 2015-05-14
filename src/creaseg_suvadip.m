function [seg,evolved_phi,its] = creaseg_suvadip(Img,init_mask,max_its,length_term,poly_degree,l2_reg_term,color,thresh)
%CREASEG_SUVADIP: Version 1 of Smooth Basis Chan Vese 

%------------------------------------------------------------------------
% Legendre Basis Chan Vese
%
% seg = region_seg(I,init_mask,max_its,alpha,display)
%
% Inputs: I           2D image
%         mask        Initialization (1 = foreground, 0 = bg)
%         max_its     Number of iterations to run segmentation for
%         length_term (optional) Weight of smoothing term
%                       higer = smoother.  default = 0.2
%         poly_degree Degree of the Legendre basis polynomial  
%         display     (optional) displays intermediate outputs
%                       default = true
%
% Outputs: seg        Final segmentation mask (1=fg, 0=bg)
%
% Description: The internal and external intensities are modeled as linear
% combination of legendre basis function, making it more adaptive to
% intensity inhomogeneity. Furthermore, the basis coefficients can be
% obtained by a close form solution, which makes it computationally
% feasible

phi = computeSDF(init_mask);
[~,vect_Bases_ortho,Bases_2D]  = LegendreBasis2D(Img,poly_degree);

param.basis_2D          = Bases_2D;
param.basis_vect        = vect_Bases_ortho;
param.init_phi          = phi;
param.Img               = im2graydouble(Img);

param.num_iter          = max_its;
param.convg_error       = thresh;
param.length_term       = length_term;
param.display_intrvl    = 30;

param.lambda1           = 1;
param.lambda2           = 1;
param.l2_reg_term       = l2_reg_term;

param.convg_count       = 20;
param.contour_color     = color;

[evolved_phi,its] = ChanVeseLegendre2D(param);

seg = evolved_phi >= 0;
evolved_phi = -evolved_phi;     % Return the negative following Creaseg's convention

end


function [SDF] = computeSDF(bwI)
%COMPUTESDF Create the signed distance function from the binary image
% inside >= 0, outside < 0

phi = bwdist(bwI)-bwdist(1-bwI)+im2double(bwI)-.5;
SDF = -phi ;

end


function [phi,its] = ChanVeseLegendre2D(opt)
%CHANVESE Segmentation by Chan Vese Legendre 2D polynomial basis 

Dirac_global            = @(x,e) ((e/pi)./(e^2.+ x.^2));
Heaviside               = @(y,e) (0.5*(1+(2/pi)*atan(y/e)));

its         = 0;
max_iter    = opt.num_iter;
u           = opt.Img;
phi         = opt.init_phi;
convg_err   = opt.convg_error;
reg_term    = opt.length_term;
l1          = opt.lambda1;
l2          = opt.lambda2;

display     = opt.display_intrvl;
color       = opt.contour_color;
count_lim   = opt.convg_count;

B           = opt.basis_2D;               % cell of 2D basis functions

lambda_l2  = opt.l2_reg_term;             % constant for L2 regularization to make matrix invertible

stop = 0;
count = 0;

fig = findobj(0,'tag','creaseg');
ud = get(fig,'userdata');

while (its < max_iter && stop == 0)
    
    h_phi           = Heaviside(phi,2);
    inside_mask     = h_phi;
    outside_mask    = 1-h_phi;
    
    [~,p1] = legendreCoef2D(u,B,inside_mask,lambda_l2);
    [~,p2] = legendreCoef2D(u,B,outside_mask,lambda_l2);
  
    curvature   = curvature_central(phi);
    delta_phi   = Dirac_global(phi,2);
    
    evolve_force = delta_phi.*(-l1*(u-p1).^2 + l2*(u-p2).^2);
    
    reg_force    = reg_term*curvature;
    
    dphi_dt = evolve_force./(max(abs(evolve_force(:)))+eps) + reg_force;
    delta_t = .5/(max(abs(dphi_dt(:)))+eps);          % Step size using CFL
    
    prev_mask = phi >=0;
    
    phi = phi + delta_t*dphi_dt;
    phi = SussmanReinitLS(phi,0.5);
    phi = NeumannBoundCond(phi);
    
    if display > 0
        if mod(its,display) == 0
            set(ud.txtInfo1,'string',sprintf('iteration: %d',its),'color',[1 1 0]);
            showCurveAndPhi(phi,ud,color);
            drawnow;
        end
    end
    
    curr_mask = phi >=0 ;
    
    count = convergence(prev_mask,curr_mask,convg_err,count);
    % count how many succesive times we have attained convergence, reduce local minima
    if count <= count_lim
        its = its + 1;
    else
        stop = 1;
    end
    
end
showCurveAndPhi(phi,ud,color); 

end


% Compute the coefficients for the legendre basis
function [coef,p] = legendreCoef2D(u, B, m, lambda)
% B: set of 2D bases, m: the inside/outside mask, lambda: l2 constraint
% c : coeffieients, p: reconstructed surface

[nrow,ncol] = size(B);
N = nrow*ncol;
C = B(:);       % The 2D bases, N of them
% Solve: KC = P;
K = zeros(N);
P = zeros(N,1);

for ii = 1 : N
   A1 = C{ii};
   tmp = A1.*u.*m;
   P(ii,1) = sum(tmp(:));           % P is a vector of length = length(C)
   for jj = 1 : N
       A2 = C{jj}.*m;
       tmp = A1.*A2;
       K(ii,jj) = sum(tmp(:));
   end
end
II = eye(size(K));
coef = (K + lambda*II)\P;

p = 0;
for ii = 1 : N
   p = p + coef(ii)*C{ii}; 
end

end



% Convergence Test
function c = convergence(p_mask,n_mask,thresh,c)
    diff = p_mask - n_mask;
    n_diff = sum(abs(diff(:)));
    if n_diff < thresh
        c = c + 1;
    else
        c = 0;
    end
end


% Compute curvature    
function k = curvature_central(u)                       

    [ux,uy] = gradient(u);                                  
    normDu = sqrt(ux.^2+uy.^2+1e-10);	% the norm of the gradient plus a small possitive number 
                                        % to avoid division by zero in the following computation.
    Nx = ux./normDu;                                       
    Ny = uy./normDu;
    nxx = gradient(Nx);                              
    [~,nyy] = gradient(Ny);                              
    k = nxx+nyy;                        % compute divergence
end


% Check boundary condition
function g = NeumannBoundCond(f)
    
    [nrow,ncol] = size(f);
    g = f;
    g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
    g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
    g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  
end





% -- Generate the orthonormal 2D Legendre basis
function [B,orthonormal_B,B_2D] = LegendreBasis2D(Img,k)
%LEGENDREBASIS compute K shifted legendre basis for the vectorized image

[Nr,Nc] = size(Img);
N = length(Img(:));     % Vectorized image

B = zeros(N,(k+1)*(k+1));
orthonormal_B = B;

[B_r,B_r_ortho] = legendre_1D(Nr,k);
[B_c,B_c_ortho] = legendre_1D(Nc,k);

% Compute the 2D bases via outer product

idx = 0;
for ii = 1 : k+1
    for jj = 1 : k+1
        idx = idx + 1;
        b1 = B_r(:,ii);
        b2 = B_c(:,jj);
        b  = b1*b2';    % outer product
        if size(b,1) ~= Nr
           b = resize(b,size(Img)); 
        end
        B_2D{ii,jj} = b;
        B(:,idx) = b(:);
        orthonormal_B(:,idx) = b(:)/max(b(:));
    end
end

end


function [B,orthonormal_B] = legendre_1D(N,k)

X = -1:2/(N-1):1;
p0 = ones(1,N);


B = zeros(N,k+1);
orthonormal_B = B;
B(:,1) = p0';
orthonormal_B(:,1) = B(:,1)/norm(B(:,1));

for ii = 2 : k+1
    Pn = 0;
    n = ii-1;   % degree
    for k = 0 : n
       Pn = Pn +  (nchoosek(n,k)^2)*(((X-1).^(n-k)).*(X+1).^k);
    end
    B(:,ii) = Pn'/(2)^n;
    orthonormal_B(:,ii) = B(:,ii)/norm(B(:,ii));
end

end



function [D] = SussmanReinitLS(D,dt)
%SUSSMANREINITLS Reinitialize LSF by Sussman reinitialization method
%D  : level set function
%dt : small timestep ~ 0.5
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
end

function shift = shiftD(M)
    shift = shiftR(M')';
end

function shift = shiftL(M)
    shift = [ M(:,2:size(M,2)) M(:,size(M,2)) ];
end

function shift = shiftR(M)
    shift = [ M(:,1) M(:,1:size(M,2)-1) ];
end

function shift = shiftU(M)
    shift = shiftL(M')';
end

function S = sussman_sign(D)
    S = D ./ sqrt(D.^2 + 1);    
end
    

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
end



function img = im2graydouble(img)    
    [dimy, dimx, c] = size(img);
    if(isfloat(img)) % image is a double
        if(c==3) 
            img = rgb2gray(uint8(img)); 
        end
    else           % image is a int
        if(c==3) 
            img = rgb2gray(img); 
        end
        img = double(img);
    end
end

