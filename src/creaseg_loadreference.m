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


function creaseg_loadreference(varargin)

    if nargin == 1
        fig = varargin{1};
    else
        fig = gcbf;
    end
    ud = get(fig,'userdata');    
    
    %-- clean figure screen
    set(ud.txtInfo1,'string','');
    set(ud.txtInfo2,'string','');
    set(ud.txtInfo3,'string','');
    set(ud.txtInfo4,'string','');
    set(ud.txtInfo5,'string',''); 

    %-- Put the "Create" button in brighter
    set(ud.handleAlgoComparison(17),'BackgroundColor',[240/255 173/255 105/255]);
    
    %-- in case, enable drawing, run and pointer buttons
    set(ud.buttonAction(1),'enable','on');
    set(ud.buttonAction(2),'enable','on');
    set(ud.buttonAction(3),'enable','on');
    
    
    %-- clean overlays and update fd structure
    keepLS = 1;
    creaseg_cleanOverlays(keepLS);
    fd = get(ud.imageId,'userdata');

    %-- flush reference strucuture if any
    fd.handleReference{1} = 0;
    
    %-- Set drawingReferenceFlag flag to 0
    fd.drawingReferenceFlag = 0;
    
    %-- Set pointsRef to empty
    fd.pointsRef = [];
    
    %-- cancel drawing mode
    for k=3:size(ud.handleInit,1)
        set(ud.handleInit(k),'BackgroundColor',[240/255 173/255 105/255]);
    end
    %-- put run button to nonselected
    set(ud.buttonAction(2),'background',[240/255 173/255 105/255]);
    %--
    set(ud.gcf,'WindowButtonDownFcn','');
    set(ud.gcf,'WindowButtonUpFcn','');      
        
    %--
    [fname,pname] = uigetfile('*.mat','Pick a file','multiselect','off','data/Reference');
    input_file = fullfile(pname,fname);
    if ~exist(input_file,'file')
        warning(['File: ' input_file ' does not exist']);
        return;
    end
    try
        junk = load(input_file);
        reference = junk.refLSF;
        clear junk;
    catch
        warning(['Could not load: ' input_file]);
        return;
    end
    
    %-- Check if the reference size is correct
    if (size(fd.data,1)~=size(reference,1)) || (size(fd.data,2)~=size(reference,2)) 
        fd.reference = [];
        set(ud.txtInfo1,'string','Error:Image and Reference must be of the same size','color', [1 0 0]);
    else
        fd.reference = reference;
        ud.LastPlot = 'reference';
        fd.method = 'Reference';
        %--
        color = ud.colorSpec(get(ud.handleContourColor,'userdata'));
        show_ref(fd.reference,ud,color);        
        set(ud.handleAlgoComparison(24),'Enable','off');
    end

    %-- UPDATE FD AND UD STRUCTURES ATTACHED TO IMAGEID AND FIG HANDLES
    set(ud.imageId,'userdata',fd);
    set(fig,'userdata',ud);

   

function show_ref(ref,ud,cl)

	hold on; [c,h] = contour(ref,[0 0],cl{1},'Linewidth',3); hold off;
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
