
% ------------------------------------------------------------
% Local Dimming
% trying to build a local dimming algorithm
% probably will be shit
% ------------------------------------------------------------

% %-------------------------------------------------------
% start
    
    clc; clearvars; close all;
    timer_total = tic;
    timer_prepare = tic;
    
% %-------------------------------------------------------
% hardware limits
    
    min_trans = 0.05;
    max_trans = 1;
    max_bklight_multiplier = 2;
    bit_depth = 8;
    
% %-------------------------------------------------------
% test parameters
    
    test_num = 7;
    zones_X = 32;
    zones_Y = 24;
    zones_size = 50;     % zones are square
    res_X = zones_X * zones_size;
    res_Y = zones_Y * zones_size;
    
% %-------------------------------------------------------
% method selection
    
    bklight_calculation_method = 'extended';
    % 'none', 'max_gray', 'max_rgb', 'avg', 'sqrt', 'corr', 'cdf_thres', 'extended', 'mix_extended'
    bklight_simulation_method = 'gauss';
    % 'none', 'gauss', 'conv', 'bma'
    cdf_threshold = 0.95;
    cdf_threshold_L = 0.4;
    cdf_threshold_H = 0.8;
    
% %-------------------------------------------------------
% blacklight simulation
    
    % blur-mask approach
    filter_center = 1;
    filter_edge = 0.25;
    filter_corner = 0.1;
    filter_size = 5;
    blur_steps = 5;
    % gauss filter approach
    bklight_gauss_diffusion = 1/2 * zones_size;
    % convolution approach
    bklight_conv_size =  1/2 * zones_size;
    
% %-------------------------------------------------------
% image input
    
    input = imread( [ 'input\' num2str(test_num) '_input.png' ] );
    % normalize
    input = double(input);
    input = input / (2^bit_depth);
    input = imresize( input, [res_Y,res_X], 'bilinear' );
    input_extend = padarray( input, [zones_size,zones_size], 'replicate','both' );
    % gpu array?
    if gpuDeviceCount > 0
        % input = gpuArray(input);
    end
    
% %-------------------------------------------------------
% direct display output
    
    direct_display = interp1( [0 1], [min_trans max_trans], input, 'linear' );
    
    time_prepare = toc(timer_prepare);
    
% %-------------------------------------------------------
% calculate backlight
    
    timer_bklight_calc = tic;
    
    backlight_index = zeros(zones_Y,zones_X);
    for i = 1 : zones_Y
        top = fix( res_Y / zones_Y * (i-1) + 1 );
        bottom = fix( res_Y / zones_Y * i );
        for j = 1 : zones_X
            left = fix( res_X / zones_X * (j-1) + 1 );
            right = fix( res_X / zones_X * j );
            % method select
            switch bklight_calculation_method
                case 'none'
                    backlight_index(i,j) = 1;
                case 'max_rgb'
                    zone_img = input(top:bottom,left:right,:);
                    backlight_index(i,j) = max( max( max( zone_img ) ) );
                case 'max_gray'
                    zone_intensity = im2gray( input(top:bottom,left:right,:) );
                    backlight_index(i,j) = max( max( zone_intensity ) );
                case 'avg'
                    zone_pixel_count = ( bottom - top + 1 ) * ( right - left + 1 );
                    zone_intensity = im2gray( input(top:bottom,left:right,:) );
                    zone_sum = sum( sum( zone_intensity ) );
                    backlight_index(i,j) = zone_sum / zone_pixel_count;
                case 'sqrt'
                    zone_pixel_count = ( bottom - top + 1 ) * ( right - left + 1 );
                    zone_intensity = im2gray( input(top:bottom,left:right,:) );
                    zone_sum = sum( sum( zone_intensity ) );
                    backlight_index(i,j) = sqrt( zone_sum / ( zone_pixel_count ) );
                case 'corr'
                    zone_pixel_count = ( bottom - top + 1 ) * ( right - left + 1 );
                    zone_intensity = im2gray( input(top:bottom,left:right,:) );
                    zone_max_gray = max( max( zone_intensity ) );
                    zone_sum = sum( sum( zone_intensity ) );
                    zone_avg = zone_sum / zone_pixel_count;
                    zone_diff = zone_max_gray - zone_avg;
                    backlight_index(i,j) = zone_avg + 0.5 * ( zone_diff + (zone_diff^2) / (2^zone_pixel_count) );
                case 'cdf_thres'
                    zone_pixel_count = ( bottom - top + 1 ) * ( right - left + 1 );
                    zone_intensity = im2gray( input(top:bottom,left:right,:) );
                    [zone_pdf,] = imhist(zone_intensity);
                    zone_cdf = zeros( 1, (2^bit_depth) );
                    zone_cdf(1) = zone_pdf(1);
                    for k = 2 : (2^bit_depth)
                        zone_cdf(k) = zone_cdf(k-1) + zone_pdf(k);
                    end
                    zone_cdf = interp1([0 zone_pixel_count], [0 1], zone_cdf, 'linear');
                    for k = 1 : (2^bit_depth)
                        if zone_cdf(k) >= cdf_threshold
                            break;
                        end
                    end
                    backlight_index(i,j) = k / (2^bit_depth);
                    clearvars k;
                case 'extended'
                    zone_pixel_count = ( bottom - top + 1 + 2*round(zones_size/2) ) * ( right - left + 1 + 2*round(zones_size/2) );
                    zone_img = input_extend( top+round(zones_size/2) : bottom+3*round(zones_size/2), left+round(zones_size/2) : right+3*round(zones_size/2), : );
                    backlight_index(i,j) = max( max( max( zone_img ) ) );
                case 'mix_extended'
                    zone_pixel_count = ( bottom - top + 1 + 2*round(zones_size/2) ) * ( right - left + 1 + 2*round(zones_size/2) );
                    zone_intensity = im2gray( input_extend( top+round(zones_size/2) : bottom+3*round(zones_size/2), left+round(zones_size/2) : right+3*round(zones_size/2), : ) );
                    zone_max_gray = max( max( zone_intensity ) );
                    zone_sum = sum( sum( zone_intensity ) );
                    zone_avg = zone_sum / zone_pixel_count;
                    zone_diff = zone_max_gray - zone_avg;
                    backlight_index(i,j) = zone_avg + 0.5 * ( zone_diff + (zone_diff^2) / (2^zone_pixel_count) );
                    zone_img = input(top:bottom,left:right,:);
                    backlight_index(i,j) = max( max( max( max( zone_img ) ) ), backlight_index(i,j) );
                otherwise
                    backlight_index(i,j) = 1;
            end
            backlight_index(i,j) = imdiscretize( backlight_index(i,j), bit_depth );
            clearvars zone_*;
        end
    end
    clearvars i j top bottom left right;
    
% %-------------------------------------------------------
% smooth out backlight pattern
    
    backlight_index_smooth = backlight_index;
    for i = 1 : zones_Y
        for j = 1 : zones_X
            surrounding_bklight = zeros(4,1);
            if i < zones_Y
                surrounding_bklight(1) = backlight_index(i+1,j);
            end
            if i > 1
                surrounding_bklight(2) = backlight_index(i-1,j);
            end
            if j < zones_X
                surrounding_bklight(3) = backlight_index(i,j+1);
            end
            if j > 1
                surrounding_bklight(4) = backlight_index(i,j-1);
            end
            if backlight_index_smooth(i,j) < max(surrounding_bklight) - 0.5
                backlight_index_smooth(i,j) = ( backlight_index_smooth(i,j) + max(surrounding_bklight) ) / 2;
            end
            clearvars surrounding_bklight;
        end
    end
    clearvars i j;
    if ~strcmp( bklight_calculation_method, 'extended' )
        backlight_index = backlight_index_smooth;
    end
    
% %-------------------------------------------------------
% backlight border extend
    
    backlight_index_extend = padarray( backlight_index, [1,1], 'replicate', 'both' );
    
    % get the full-res backlight pattern
    backlight_raw = imresize( backlight_index, zones_size, 'box' );
    backlight_raw_extend = imresize( backlight_index_extend, zones_size, 'box' );
    
    time_bklight_calc = toc(timer_bklight_calc);
    
% %-------------------------------------------------------
% simulate backlight
    
    timer_bklight_sim = tic;
    
    % method select    
    switch bklight_simulation_method
        case 'none'
            backlight_display = backlight_raw_extend;
        case 'gauss'
            backlight_display = imgaussfilt( backlight_raw_extend, bklight_gauss_diffusion, 'Padding', 0 );
        case 'conv'
            x = linspace( -bklight_conv_size*2, bklight_conv_size*2, bklight_conv_size*2*2 + 1 );
            y = linspace( -bklight_conv_size*2, bklight_conv_size*2, bklight_conv_size*2*2 + 1 );
            [X,Y] = meshgrid(x,y);
            bklight_diffusion_pattern = exp( - ( X.^2 +  Y.^2 ) / ( 2 * bklight_conv_size^2 ) );    % gauss beam as a temporary solution
            bklight_diffusion_pattern = bklight_diffusion_pattern / sum( sum( bklight_diffusion_pattern ) );    % normalize
            % backlight_display = conv2( backlight_raw_extend, bklight_diffusion_pattern, 'same' );
            backlight_display = imfilter( backlight_raw_extend, bklight_diffusion_pattern, 'same', 'conv' );    % seems to be the same, but way faster
            clearvars x y X Y;
        case 'bma'
            blur_LPF = filter_center * ones( filter_size, filter_size );
            blur_LPF( [1 filter_size] , : ) = filter_edge;
            blur_LPF( : , [1 filter_size] ) = filter_edge;
            blur_LPF( [1 filter_size] , [1 filter_size] ) = filter_corner;
            blur_LPF = blur_LPF / sum( sum( blur_LPF ) );	% normalize
            backlight_bma = backlight_index_extend;
            for i = 1 : blur_steps
                backlight_bma = imresize( backlight_bma, 2 );
                backlight_bma = imfilter( backlight_bma, blur_LPF, 0, 'same', 'conv' );
            end
            clearvars i
            backlight_display = imresize( backlight_bma, [res_Y+2*res_Y/zones_Y,res_X+2*res_X/zones_X], 'bilinear' );
        otherwise
            backlight_display = backlight_raw_extend;
    end
    % cut to final size
    backlight_display = backlight_display( res_Y/zones_Y + 1 : res_Y + res_Y/zones_Y, res_X/zones_X + 1 : res_X + res_X/zones_X );
    
    time_bklight_sim = toc(timer_bklight_sim);
    
% %-------------------------------------------------------
% calculate input image over dimmed backlight
    
    timer_compensate = tic;
    
    uncalibrated_output = backlight_display .* direct_display;
    image_diff = input - uncalibrated_output;
    
% %-------------------------------------------------------
% calculate overlay
    
    % backlight scale-up
    if strcmp( bklight_calculation_method, 'none' )
        bklight_enhance_scale = 1.0;
    elseif strcmp( bklight_calculation_method, 'extended' )
        bklight_enhance_scale = 1.05;
    elseif strcmp( bklight_calculation_method, 'mix_extended' )
        bklight_enhance_scale = 1.1;
    elseif strcmp( bklight_calculation_method, 'max_rgb' ) || strcmp( bklight_calculation_method, 'max_gray' )
        bklight_enhance_scale = 1.2;
    else
        bklight_enhance_scale = 1.35;
    end
    backlight_index = backlight_index * bklight_enhance_scale;
    backlight_display = backlight_display * bklight_enhance_scale;
    % overlay image
    calibrated_input = input ./ backlight_display;
    calibrated_input = fillmissing( calibrated_input, 'constant', 0 );      % possible NaN results due to 0 bklight
    calibrated_input = interp1( [min_trans max_trans], [0 1], calibrated_input, 'linear', 'extrap' );
    calibrated_input = max( calibrated_input, 0 );       % cap the value to remove possible NaN results
    calibrated_input = min( calibrated_input, 1 );
    overlay_display = imdiscretize( calibrated_input, bit_depth );
    overlay_display = interp1( [0 1], [min_trans max_trans], overlay_display, 'linear' );   % possible NaN results here, need a cap on input
    
    time_compensate = toc(timer_compensate);
    
% %-------------------------------------------------------
% simulate outcome
    
    timer_result_disp = tic;
    
    % reference backlight distribution
    backlight_reference = imgaussfilt( backlight_raw_extend * bklight_enhance_scale, bklight_gauss_diffusion, 'Padding', 0 );
    backlight_reference = backlight_reference( res_Y/zones_Y + 1 : res_Y + res_Y/zones_Y, res_X/zones_X + 1 : res_X + res_X/zones_X );
    % simulate outcome
    outcome = backlight_reference .* overlay_display;
    error = outcome - input;
    % error value calc
    fprintf( '> Overall error: R=%0.5f%%, G=%0.5f%%, B=%0.5f%%, Total= +%0.5f%%/-%0.5f%%\n', sum(sum(abs(100*error)))/(res_X*res_Y), sum(sum(100*(im2gray(error))))/(res_X*res_Y), sum(sum(100*(im2gray(-error))))/(res_X*res_Y) );
    fprintf( '     instead of: R=%0.5f%%, G=%0.5f%%, B=%0.5f%%, Total= +%0.5f%%/-%0.5f%%\n', sum(sum(abs(100*(direct_display-input))))/(res_X*res_Y), sum(sum(100*(im2gray(direct_display-input))))/(res_X*res_Y), sum(sum(100*(im2gray(input-direct_display))))/(res_X*res_Y) );
    fprintf( '> Backlight reduction: %0.5f instead of 1\n', sum(sum(backlight_index))/(zones_X*zones_Y) );
    
% %-------------------------------------------------------
% result display
    
    figure('Name','Result','NumberTitle','off','WindowState','maximized','Color','black')
    % input
    subplot(2,3,2);
	imshow(input)
	title( 'Input','Color','white' );
    axis image
    % direct display
    subplot(2,3,1);
	imshow(direct_display)
	title( 'Direct Display','Color','white' );
    axis image
    % output
    subplot(2,3,3);
	imshow(outcome)
	title( 'Dimmed Display','Color','white' );
    axis image
    % overlay
    subplot(2,3,5);
	imshow(overlay_display)
	title( 'Overlay','Color','white' );
    axis image
    % overlay display
    subplot(2,3,4);
	imshow(0.5+10*error)
	title( 'Total Error (10x)','Color','white' );
    axis image
    % backlight
    subplot(2,3,6);
	imshow(backlight_display,[0 max(max(backlight_display))])
	title( 'Backlight Sim','Color','white' );
    axis image
    
% %-------------------------------------------------------
% end
    
    time_result_disp = toc(timer_result_disp);
    time_total = toc(timer_total);
    
    fprintf( '\n> 分段耗时: ' );
    fprintf( 'prepare:%0.4fs, ', time_prepare );
    fprintf( 'bl-calc:%0.4fs, ', time_bklight_calc );
    fprintf( 'bl-sim:%0.4fs, ', time_bklight_sim );
    fprintf( 'compensate:%0.4fs, ', time_compensate );
    fprintf( 'result-disp:%0.4fs ', time_result_disp );
    fprintf( '\n> 总计耗时: %0.4fs\n', time_total );
    
    clearvars timer_*;
    
    