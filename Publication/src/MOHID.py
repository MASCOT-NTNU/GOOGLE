"""
MOHID handles the data preparation for the operation day. It imports Delft3D data and
krige the updated field based on the forecast data from MOHID data source.
"""
from WGS import WGS
from Config import Config
import os
import numpy as np
import h5py
from shapely.geometry import Point


class MOHID:
    """ Load setup. """
    __setup = Config()
    __mission_date = __setup.get_mission_date()
    __clock_start = __setup.get_clock_start()
    __clock_end = __setup.get_clock_end()
    __polygon_operational_area_shapely = __setup.get_polygon_operational_area_shapely()

    """ MOHID data manipulation. """
    __folderpath_mohid = os.getcwd() + "/../../../../Data/Porto/OASIS/mohid/"
    __files_mohid = os.listdir(__folderpath_mohid); __files_mohid.sort()
    __ind_date = __files_mohid.index(__mission_date)
    __datapath_mohid = __folderpath_mohid + __files_mohid[__ind_date] + "/WaterProperties.hdf5"
    __data_mohid = h5py.File(__datapath_mohid, 'r')
    __grid_mohid = __data_mohid.get('Grid')
    __lat_mohid = np.array(__grid_mohid.get("Latitude"))[:-1, :-1].flatten()
    __lon_mohid = np.array(__grid_mohid.get("Longitude"))[:-1, :-1].flatten()
    __depth_mohid = []
    __salinity_mohid = []
    for i in range(1, 26):
        string_z = "Vertical_{:05d}".format(i)
        string_sal = "salinity_{:05d}".format(i)
        __depth_mohid.append(np.mean(np.array(__grid_mohid.get("VerticalZ").get(string_z)), axis=0))
        __salinity_mohid.append(np.mean(np.array(__data_mohid.get("Results").get("salinity").get(string_sal)), axis=0))
    __depth_mohid = np.array(__depth_mohid)
    __salinity_mohid = np.array(__salinity_mohid)

    # Filter outbound data
    __filter_mohid = []
    for i in range(len(__lat_mohid)):
        __filter_mohid.append(__polygon_operational_area_shapely.contains(Point(__lat_mohid[i], __lon_mohid[i])))
    __ind_legal_mohid = np.where(__filter_mohid)[0]

    __salinity_mohid_time_ave = np.mean(__salinity_mohid[__clock_start:__clock_end, :, :], axis=0).flatten()[
        __ind_legal_mohid]
    xm, ym = WGS.latlon2xy(__lat_mohid[__ind_legal_mohid], __lon_mohid[__ind_legal_mohid])
    __dataset_mohid = np.stack((xm, ym, __salinity_mohid_time_ave), axis=1)

    @staticmethod
    def get_dataset() -> np.ndarray:
        """ Return dataset of Delft3D.
        Example:
             dataset = np.array([[lat, lon, salinity]])
        """
        return MOHID.__dataset_mohid


if __name__ == "__main__":
    m = MOHID()


