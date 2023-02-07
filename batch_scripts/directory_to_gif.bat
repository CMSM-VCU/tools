:: Convert one or more directories containing png images to gifs. Each directory
::     is converted to one gif.
:: In Windows Explorer, the folder(s) can be dragged onto this file directly.
::
:: REQUIRES ImageMagick (imagemagick.org)
::
:: Resulting gif is saved in parent directory with same name as image directory
:: e.g.:
:: project/
::   images_1/
::     001.png
::     002.png
::   images_1.gif  <- Creates this file
::
:: Images are read in alphanumeric order, so leading zeroes are required for correct ordering.
:: e.g.
:: Bad:  6.png, 32.png, 104.png    - The frames will be in the order 104->32->6
:: Good: 006.png, 032.png, 104.png - The frames will be in the order 6->32->104

@echo off

:: Duration of each frame in centiseconds (100 = 1 second per frame)
SET frame_duration=10

setlocal enabledelayedexpansion

for %%x in (%*) do (
    echo Converting %%~x ...
    magick convert -delay !frame_duration! -layers OptimizeTransparency %%~nx\*.png %%~nx.gif
    echo Created %%~nx.gif
    echo:
)

echo:
echo Finished
timeout -t 3
