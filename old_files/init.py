from CNN_from_Scratch_Better_Approach import train

hour = 60 ** 2

# netMax, loss_tracker = train(2 * hour, 5, 20, False, True, False,title="P0 K1 Transposed2D noTorchMax
# lr0.05 BCEweighted")
_, _ = train(2 * hour, 10, 10, True, False, False,"")

# @todo Analyse der Ergebnisse
# @todo Implement Average und Interpolate ODER nur Average mit Image Skalierung und dem Verweis, dass\
#  es aus technischer Sicht nicht möglich ist mit meinem PC

colorsys = False
if colorsys:
    import colorsys


    def scale_lightness(rgb, scale_l):
        # convert rgb to hls
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # manipulate h, l, s values and return as rgb
        return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)