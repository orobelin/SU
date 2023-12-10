Hello Olivier!

I'm sorry that these data are not more sorted. I simply don't have the time to make them nicer right now. I hope this helps a bit at least!

Everything except the Koblet et al. (2007) data and the 1910 stuff is unpublished and made by me.

Here's a short description of this data dump:

## Shapefiles

- `glacier.geojson`: A 1910 and 2021 outline. The 1910 shape may be newer than in the other file, so use that.
- `glacier.gpkg`: Outlines from 1910, 1959, 1969, 1980, 1990, 1999 and 2017

## DEM

- `DEM_1910.tif`: 10x10 DEM from late July 1910 is described by Holmlund and Holmlund (2019)
- `Keb_DEM_1946_181124.tif`: A DEM from 1946. It's a very difficult dataset so the DEM is not great
- `Keb_DEM_1959_181125.tif`: A DEM from 1959. This one is quite good but may contain some gaps
- `Keb_DEM_1980high_181114.tif`: A DEM from the high-altitude images from 1980. There are also low-altitude images somewhere, but I didn't have access to them.
- `Storgl*2018*.tif`: DEM and ortho from 2018 from helicopter images. I made it quite a while ago so it may not be perfect.
- `tarfala_19*.tif`: I have no idea what these are.
- `koblet2007/`: Data from Koblet et al. (2007). These are not good at all in my opinion!
- `Tarfala_DEM_2015.tif`: DEM from the lidar scan by Lantmäteriet in 2015.
- `Tarfala_Ortho_2008.tif`: Ortho from Lantmäteriet in 2008.
- `dem_2021.tif`: DEM from helicopter images in 2021.
- `basal.tif`: A basal DEM from digitized data by Björnsson et al., (1980) and som corrections made using the 2021 DEM.
- `surface_2015.tif`: DEM provided by Jamie at the begining of Olivier's internship (maybe the same that `Tarfala_DEM_2015.tif` ?)

## New

New files created for the simulation.
- `surface_1910.nc`: Raster of 1910 DEM reproject with data out of the glacier + 1 Band for the bed (from basal.tif file)
- `surface_1959.nc`: Raster of 1959 DEM with a mask with the shape of the glacier at this date, data of 'basal.tif' out of the mask and reproject at the size of basal.tif
- `Courbes`: Altitude lines for visualization
- `thick_2015.tif`: Thickness deduced from 'surface_2015.tif' and 'basal.tif'
- `bed.tif`: Bed from 'basal.tif' with a mask of the glacier from 'surface_2015.tif' (used in the simulations)

##How to reproject the old DEM
1- Create a shapefile with the right year
2- Use "découper un raster selon une emprise" and select the shape file
3- Reshape the raster at the same resolution than the raster of the bed (basal.tif)
4- Use "fussioner" command, select the 2 files (surface with mask around the glacier and bed) and run it
