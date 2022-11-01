
% convert8bit
% convert image file (double) to 256-level (double)

function [img_out] = imdiscretize( norm_img_in, bit_depth )

    max_scale = max( max( max( norm_img_in ) ) );
    normalized_input = norm_img_in / max_scale;
    img_out = double( uint16( round( normalized_input * (2^bit_depth) ) ) ) / (2^bit_depth) * max_scale;
    
end