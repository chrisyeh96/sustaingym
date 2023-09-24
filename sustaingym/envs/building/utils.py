"""
This module primarily implements the ParameterGenerator() function which
generates the parameters dict for BuildingEnv.

All of the building layouts, U-factor values (thermal transmittance, in
W/m2-K), ground temperatures, and weather data come from the Building Energy
Codes Program: https://www.energycodes.gov/prototype-building-models.

Buildings models
- HTM files were extracted from the "Individual Standard 90.1 Prototype Building
  Models" (version 90.1-2019)
- We associate each building type with a list of 7 U-factor values (thermal
  transmittance, in W/m2-K) for different surfaces in the building in the
  order: [intwall, floor, outwall, roof, ceiling, groundfloor, window]
  These values are manually compiled from the HTM files, with missing
  values manually filled in from similar building types. For example,
  values missing from OfficeSmall were filled in from OfficeMedium.

Monthly ground temperature values come from the
"Site:GroundTemperature:FCfactorMethod" table in the building HTM files.

Weather data come from EnergyPlus TMY3 Weather Files (in *.epw format)
also provided by the Building Energy Codes Program.
"""
from __future__ import annotations

from collections.abc import Sequence
from collections import defaultdict
import os
from typing import Any, NamedTuple

import numpy as np
import pvlib
from scipy import interpolate

from sustaingym.data.utils import read_to_stringio


BUILDINGS = {
    "ApartmentHighRise": (
        "ASHRAE901_ApartmentHighRise_STD2019_Tucson.table.htm",
        [6.299, 3.285, 0.384, 0.228, 3.839, 0.287, 2.786],
    ),
    "ApartmentMidRise": (
        "ASHRAE901_ApartmentMidRise_STD2019_Tucson.table.htm",
        [6.299, 3.285, 0.384, 0.228, 3.839, 0.287, 2.786],
    ),
    "Hospital": (
        "ASHRAE901_Hospital_STD2019_Tucson.table.htm",
        [6.299, 3.839, 0.984, 0.228, 3.839, 3.285, 2.615],
    ),
    "HotelLarge": (
        "ASHRAE901_HotelLarge_STD2019_Tucson.table.htm",
        [6.299, 0.228, 0.984, 0.228, 0.228, 2.705, 2.615],
    ),
    "HotelSmall": (
        "ASHRAE901_HotelSmall_STD2019_Tucson.table.htm",
        [6.299, 3.839, 0.514, 0.228, 3.839, 0.1573, 2.615],
    ),
    "OfficeLarge": (
        "ASHRAE901_OfficeLarge_STD2019_Tucson.table.htm",
        [6.299, 3.839, 0.984, 0.228, 4.488, 3.839, 2.615],
    ),
    "OfficeMedium": (
        "ASHRAE901_OfficeMedium_STD2019_Tucson.table.htm",
        [6.299, 3.839, 0.514, 0.228, 4.488, 0.319, 2.615],
    ),
    "OfficeSmall": (
        "ASHRAE901_OfficeSmall_STD2019_Tucson.table.htm",
        [6.299, 3.839, 0.514, 0.228, 4.488, 0.319, 2.615],
    ),
    "OutPatientHealthCare": (
        "ASHRAE901_OutPatientHealthCare_STD2019_Tucson.table.htm",
        [6.299, 3.839, 0.514, 0.228, 3.839, 0.5650e-02, 2.615],
    ),
    "RestaurantFastFood": (
        "ASHRAE901_RestaurantFastFood_STD2019_Tucson.table.htm",
        [6.299, 0.158, 0.547, 4.706, 0.158, 0.350, 2.557],
    ),
    "RestaurantSitDown": (
        "ASHRAE901_RestaurantSitDown_STD2019_Tucson.table.htm",
        [6.299, 0.158, 0.514, 4.706, 0.158, 0.194, 2.557],
    ),
    "RetailStandalone": (
        "ASHRAE901_RetailStandalone_STD2019_Tucson.table.htm",
        [6.299, 0.047, 0.984, 0.228, 0.228, 0.047, 3.695],
    ),
    "RetailStripmall": (
        "ASHRAE901_RetailStripmall_STD2019_Tucson.table.htm",
        [6.299, 0.1125, 0.514, 0.228, 0.228, 0.1125, 3.695],
    ),
    "SchoolPrimary": (
        "ASHRAE901_SchoolPrimary_STD2019_Tucson.table.htm",
        [6.299, 0.144, 0.514, 0.228, 0.228, 0.144, 2.672],
    ),
    "SchoolSecondary": (
        "ASHRAE901_SchoolSecondary_STD2019_Tucson.table.htm",
        [6.299, 3.839, 0.514, 0.228, 3.839, 0.144, 2.672],
    ),
    "Warehouse": (
        "ASHRAE901_Warehouse_STD2019_Tucson.table.htm",
        [0.774, 0.1926, 1.044, 0.5892, 10.06, 0.1926, 2.557],
    ),
}

GROUND_TEMP = {
    "Albuquerque": [13.7, 7.0, 2.1, 2.6, 4.3, 8.8, 13.9, 17.8, 23.2, 25.6, 24.1, 20.5],
    "Atlanta": [16.0, 11.9, 7.7, 4.0, 7.9, 13.8, 17.2, 20.8, 24.8, 26.1, 26.5, 22.5],
    "Buffalo": [9.7, 6.0, -2.2, -3.4, -4.2, 2.7, 7.5, 13.7, 18.6, 22.0, 20.7, 16.5],
    "Denver": [7.1, 3.0, -1.0, 0.8, -0.2, 4.8, 6.1, 13.7, 22.2, 22.7, 21.7, 18.5],
    "Dubai": [29.5, 25.5, 21.1, 19.2, 20.8, 23.1, 26.5, 31.4, 33.0, 35.1, 35.3, 32.5],
    "ElPaso": [18.3, 11.2, 6.8, 8.1, 10.3, 12.5, 19.2, 23.8, 27.9, 27.5, 26.3, 23.4],
    "Fairbanks": [-3.1, 17.7, 19.3, 17.6, 15.4, 10.3, 0.7, 10.6, 16.0, 16.9, 14.2, 6.7],
    "GreatFalls": [8.6, 2.8, 4.1, 8.8, 2.2, 0.3, 6.7, 10.1, 16.5, 20.6, 19.2, 14.7],
    "HoChiMinh": [26.9, 26.7, 26.0, 26.4, 27.5, 28.3, 29.2, 29.0, 28.9, 27.2, 27.5, 27.6],
    "Honolulu": [26.2, 24.8, 23.7, 22.5, 22.8, 23.2, 23.8, 25.2, 25.9, 26.9, 27.1, 26.9],
    "InternationalFalls": [5.4, 2.0, 14.6, 16.9, 11.5, 6.2, 4.0, 13.4, 18.0, 19.7, 17.9, 12.3],
    "NewDelhi": [25.1, 19.6, 14.5, 13.4, 17.0, 22.4, 29.1, 33.0, 33.6, 31.7, 30.0, 28.7],
    "NewYork": [14.0, 7.3, 3.3, 1.2, -0.2, 5.6, 10.9, 16.1, 21.7, 25.0, 24.8, 19.9],
    "PortAngeles": [9.3, 6.7, 4.1, 4.2, 4.2, 5.9, 9.0, 10.0, 13.3, 15.0, 15.7, 13.4],
    "Rochester": [7.4, 0.0, 7.6, 12.6, 7.7, 0.3, 7.0, 14.2, 19.2, 20.9, 20.0, 15.4],
    "SanDiego": [18.8, 14.3, 13.6, 13.2, 13.3, 12.6, 15.3, 15.6, 17.7, 19.4, 19.7, 18.5],
    "Seattle": [11.4, 8.1, 5.4, 4.5, 5.8, 8.3, 10.9, 13.0, 15.6, 17.7, 18.8, 15.1],
    "Tampa": [24.2, 18.9, 15.7, 13.6, 15.5, 17.1, 21.2, 26.9, 27.6, 27.9, 27.4, 26.2],
    "Tucson": [20.9, 15.4, 11.9, 14.8, 12.7, 15.4, 23.3, 26.3, 31.2, 30.4, 29.8, 27.8],
}

WEATHER = {
    "Very_Hot_Humid": "USA_HI_Honolulu.Intl.AP.911820_TMY3.epw",
    "Hot_Humid": "USA_FL_Tampa-MacDill.AFB.747880_TMY3.epw",
    "Hot_Dry": "USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw",
    "Warm_Humid": "USA_GA_Atlanta-Hartsfield.Jackson.Intl.AP.722190_TMY3.epw",
    "Warm_Dry": "USA_TX_El.Paso.Intl.AP.722700_TMY3.epw",
    "Warm_Marine": "USA_CA_San.Deigo-Brown.Field.Muni.AP.722904_TMY3.epw",
    "Mixed_Humid": "USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw",
    "Mixed_Dry": "USA_NM_Albuquerque.Intl.Sunport.723650_TMY3.epw",
    "Mixed_Marine": "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw",
    "Cool_Humid": "USA_NY_Buffalo.Niagara.Intl.AP.725280_TMY3.epw",
    "Cool_Dry": "USA_CO_Denver-Aurora-Buckley.AFB.724695_TMY3.epw",
    "Cool_Marine": "USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw",
    "Cold_Humid": "USA_MN_Rochester.Intl.AP.726440_TMY3.epw",
    "Cold_Dry": "USA_MT_Great.Falls.Intl.AP.727750_TMY3.epw",
    "Very_Cold": "USA_MN_International.Falls.Intl.AP.727470_TMY3.epw",
    "Subarctic/Arctic": "USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw",
}


class Zone(NamedTuple):
    name: str                   # 0
    Zaxis: float                # 1
    Xmin: float                 # 2
    Xmax: float                 # 3
    Ymin: float                 # 4
    Ymax: float                 # 5
    Zmin: float                 # 6
    Zmax: float                 # 7
    FloorArea: float            # 8
    ExteriorGrossArea: float    # 9
    ExteriorWindowArea: float   # 10
    ind: int                    # 11, can't use name "index" because of tuple.index()


def get_zones(
    filename: str,
) -> tuple[list[list[Zone]], int, list[Zone]]:
    """Parses information from the HTM file and sorts each zone by layer.

    Args:
        filename: path to building HTM file.

    Returns:
        layers: Zones grouped by Zaxis, layers[i] is a list of Zones in layer i
        n: number of zones
        all_zones: list of n Zones
    """
    # Initialize lists for storing zone information
    cord: list[str | float] = []
    cordall: list[list[str | float]] = []

    # Read all lines of the html file
    with open(filename, 'r') as f:
        htmllines = f.readlines()

    # Initialize count and printflag variables
    count = 0
    printflag = False

    # Iterate through each line in the html file
    for line in htmllines:
        count += 1

        # Turn off the printflag after the 'Zone info' chart
        if 'Zone Internal Gains Nominal' in line:
            printflag = False

        # Extract information when the printflag is True
        if printflag:
            # Zone_name
            if (count - 35) % 32 == 0 and count != 3:
                cord.append(line[22:-6])
            # Zaxis
            if (count - 42) % 32 == 0 and count != 10:
                cord.append(float(line[22:-6]))
            # Xmin
            if (count - 46) % 32 == 0 and count != 14:
                cord.append(float(line[22:-6]))
            # Xmax
            if (count - 47) % 32 == 0 and count != 15:
                cord.append(float(line[22:-6]))
            # Ymin
            if (count - 48) % 32 == 0 and count != 16:
                cord.append(float(line[22:-6]))
            # Ymax
            if (count - 49) % 32 == 0 and count != 17:
                cord.append(float(line[22:-6]))
            # Zmin
            if (count - 50) % 32 == 0 and count != 18:
                cord.append(float(line[22:-6]))
            # Zmax
            if (count - 51) % 32 == 0 and count != 19:
                cord.append(float(line[22:-6]))
            # FloorArea
            if (count - 56) % 32 == 0 and count != 24:
                cord.append(float(line[22:-6]))
            # ExteriorNetArea
            if (count - 58) % 32 == 0 and count != 26:
                cord.append(float(line[22:-6]))
            # ExteriorWindowArea
            if (count - 59) % 32 == 0 and count != 27:
                cord.append(float(line[22:-6]))

                # Append the current cord to cordall and reset cord
                cordall.append(cord)
                cord = []

        # Set printflag to True when 'Zone Information' is encountered in the line
        if 'Zone Information' in line:
            printflag = True
            count = 0

    # Calculate the total number of zones
    n = len(cordall)

    # Sort cordall by Zaxis
    cordall.sort(key=lambda x: x[1])

    # Convert cordall to Zone NamedTuples
    all_zones = [Zone(*cord, ind=int(i)) for i, cord in enumerate(cordall)]

    layers = []
    current_layer = []
    current_zaxis = all_zones[0].Zaxis

    for i, zone in enumerate(all_zones):
        if zone.Zaxis == current_zaxis:
            # If Zaxis remains the same, add the zone to the current_layer
            current_layer.append(zone)
        else:
            # If the Zaxis value changes, add the current_layer to layers
            # and reset the current_layer and current_zaxis variables
            layers.append(current_layer)
            current_layer = [zone]
            current_zaxis = zone.Zaxis

        # If this is the last zone, append the current_layer to layers
        if i == len(all_zones) - 1:
            layers.append(current_layer)

    return layers, n, all_zones


def checkconnect(z1: Zone, z2: Zone) -> bool:
    """Checks whether zones in the same layer are connected."""
    z1_min_in_z2 = (
        z2.Xmin <= z1.Xmin <= z2.Xmax
        and z2.Ymin <= z1.Ymin <= z2.Ymax
    )
    z1_max_in_z2 = (
        z2.Xmin <= z1.Xmax <= z2.Xmax
        and z2.Ymin <= z1.Ymax <= z2.Ymax
    )
    return z1_min_in_z2 or z1_max_in_z2


def checkconnect_layer(z1: Zone, z2: Zone) -> bool:
    """Checks whether zones in different layers are connected."""
    z1_min_in_z2 = (
        z2.Xmin <= z1.Xmin < z2.Xmax
        and z2.Ymin <= z1.Ymin < z2.Ymax
    )
    z1_max_in_z2 = (
        z2.Xmin < z1.Xmax <= z2.Xmax
        and z2.Ymin < z1.Ymax <= z2.Ymax
    )
    return z1_min_in_z2 or z1_max_in_z2


def Nfind_neighbor(
    n: int,
    layers: Sequence[Sequence[Zone]],
    U_Wall: Sequence[float],
    SpecificHeat_avg: float,
) -> tuple[dict[str, list[int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is for the building model.

    Args:
        n: number of rooms
        layers: sorted layer list
        U_Wall: list of 7 U-values (thermal transmittance) for different
            surfaces in the building in the order
            [intwall, floor, outwall, roof, ceiling, groundfloor, window].
        SpecificHeat_avg: Specific heat.

    Returns:
        dicRoom: map dictionary for neighbour,n+1 by n,
        Rtable: shape [n, n+1]
        Ctable: shape [n]
        Windowtable: shape [n]
    """
    # Initialize Rtable, Ctable, and Windowtable
    Rtable = np.zeros((n, n + 1))
    Ctable = np.zeros(n)
    Windowtable = np.zeros(n)

    # Unpack U_Wall values
    Walltype, Floor, OutWall, OutRoof, Ceiling, _, Window = U_Wall

    # Set air density value
    Air = 1.225  # kg/m^3

    # Initialize the dictionary for room neighbors
    dicRoom = defaultdict[str, list[int]](list)
    outind = n

    # Iterate through each layer in layers
    num_layers = len(layers)
    for k, layer in enumerate(layers):
        FloorRoom_num = len(layer)

        # Check for neighbors in the layer above
        if k + 1 < num_layers:
            for z1 in layer:
                for z2 in layers[k+1]:
                    # Check if zones are connected between layers
                    if (checkconnect_layer(z1, z2) or checkconnect_layer(z2, z1)):
                        # Calculate cross-sectional area
                        x_overlap = min(z1.Xmax, z2.Xmax) - max(z1.Xmin, z2.Xmin)
                        y_overlap = min(z1.Ymax, z2.Ymax) - max(z1.Ymin, z1.Ymin)
                        crossarea = x_overlap * y_overlap

                        # Calculate heat transfer coefficient (U) for connected zones
                        U = crossarea * (Floor * Ceiling / (Floor + Ceiling))

                        # Update Rtable for connected zones
                        Rtable[z2.ind, z1.ind] = U
                        Rtable[z1.ind, z2.ind] = U

                        # Update the dictionary with connected zones
                        dicRoom[z1.name].append(z2.ind)
                        dicRoom[z2.name].append(z1.ind)

        # Calculate heat capacity (C) and window area for each zone in current layer
        for i, z1 in enumerate(layer):
            height = z1.Zmax - z1.Zmin
            xleng = z1.Xmax - z1.Xmin
            yleng = z1.Ymax - z1.Ymin

            # Compute heat capacity (C) for the current z1
            C_room = SpecificHeat_avg * height * xleng * yleng * Air
            Ctable[z1.ind] = C_room

            # Update Windowtable for the current z1
            Windowtable[z1.ind] = z1.ExteriorWindowArea

            # Update Rtable for exterior zones
            if z1.ExteriorGrossArea > 0 or (i == len(layer) - 1):
                if i == len(layer) - 1:
                    Rtable[z1.ind, -1] = (
                        z1.ExteriorGrossArea * OutWall
                        + xleng * yleng * OutRoof
                        + z1.ExteriorWindowArea * Window
                    )

                else:
                    Rtable[z1.ind, -1] = (
                        z1.ExteriorGrossArea * OutWall
                        + z1.ExteriorWindowArea * Window
                    )

                # Update the dictionary
                dicRoom[z1.name].append(outind)

            # Check for neighbors within the same layer
            for j in range(i + 1, FloorRoom_num):
                z2 = layer[j]

                # Check if zones are connected within the same layer
                if checkconnect(z1, z2) or checkconnect(z2, z1):
                    # Calculate the length of the shared wall
                    x_overlap = min(z1.Xmax, z2.Xmax) - max(z1.Xmin, z2.Xmin)
                    y_overlap = min(z1.Ymax, z2.Ymax) - max(z1.Ymin, z1.Ymin)
                    length = np.sqrt(x_overlap**2 + y_overlap**2)

                    # Calculate heat transfer coefficient (U) for connected zones
                    U = height * length * Walltype

                    # Update Rtable for connected zones
                    Rtable[z2.ind, z1.ind] = U
                    Rtable[z1.ind, z2.ind] = U

                    # Update the dictionary with connected zones
                    dicRoom[z1.name].append(z2.ind)
                    dicRoom[z2.name].append(z1.ind)

    return dicRoom, Rtable, Ctable, Windowtable


def ParameterGenerator(
    building: str,
    weather: str,
    Location: str,
    U_Wall: Sequence[float] = (0,) * 7,
    Ground_Tp: Sequence[float] = (0,) * 12,
    shgc: float = 0.252,
    shgc_weight: float = 0.01,
    ground_weight: float = 0.5,
    full_occ: np.ndarray | float = 0,
    max_power: np.ndarray | int = 8000,
    AC_map: np.ndarray | int = 1,
    time_reso: int = 3600,
    reward_gamma: tuple[float, float] = (0.001, 0.9990),
    target: np.ndarray | float = 22,
    activity_sch: np.ndarray = np.ones(100000000) * 1 * 120,
    temp_range: tuple[float, float] = (-40, 40),
    is_continuous_action: bool = True,
    root: str = ''
) -> dict[str, Any]:
    """Generates parameters from the selected building and temperature file for the env.

    Args:
        building: name of a building in `BUILDINGS`, or path (relative to ``root``)
            to a htm file for building idf
        weather: name of a weather condition in `WEATHER`, or path to an epw file.
        Location: name of a location in `GROUND_TEMP`
        U_Wall: list of 7 U-values (thermal transmittance) for different
            surfaces in the building in the order
            [intwall, floor, outwall, roof, ceiling, groundfloor, window].
            Only used if ``building`` cannot be found in `BUILDINGS`
        Ground_Tp: Monthly ground temperature (in Celsius) when location is not in `GROUND_TEMP`.
        shgc: Solar Heat Gain Coefficient for the window (unitless)
        shgc_weight: Weight factor for extra loss of solar irradiance (ghi). Default is 0.01.
        ground_weight: Weight factor for extra loss of heat from ground. Default is 0.5.
        full_occ (numpy array): Shape (n,1). Max number of people that can occupy each room. Default is all zeros.
        max_power (int): Maximum power output of a single HVAC unit, in watts. Default is 8000.
        AC_map: int, indicating presence (1) or absence (0) of AC in all rooms,
            or boolean array of shape (n,) to specify AC presence in individual
            rooms
        time_reso (int): Length of 1 timestep in seconds. Default is 3600 (1 hour).
        reward_gamma (list of two floats): [Energy penalty, temperature error penalty]. Default is [0.001,0.999].
        target (float or numpy array): Shape (n,). Target temperature setpoints for each zone. Default is 22 degrees Celsius.
        activity_sch: shape (length of the simulation,). Activity schedule of people in the building in watts/person. Default is all 120.
        temp_range: (Min temperature, max temperature) in Celcius, defining comfort range
        is_continuous_action: whether to use continuous action space (as opposed to MultiDiscrete)
        root: root directory for building and weather data files

    Returns:
        Parameter: Contains all parameters needed for environment initialization.
    """
    # Define dictionaries for building, Ground Temperature, and weather

    # Check if Location is in GROUND_TEMP, otherwise use Ground_Tp as city
    if Location not in GROUND_TEMP:
        city = Ground_Tp
    else:
        city = GROUND_TEMP[Location]

    # Calculate ground temperature for each month
    groundtemp = np.concatenate(
        [
            np.ones(31 * 24 * 3600 // time_reso) * city[0],
            np.ones(28 * 24 * 3600 // time_reso) * city[1],
            np.ones(31 * 24 * 3600 // time_reso) * city[2],
            np.ones(30 * 24 * 3600 // time_reso) * city[3],
            np.ones(31 * 24 * 3600 // time_reso) * city[4],
            np.ones(30 * 24 * 3600 // time_reso) * city[5],
            np.ones(31 * 24 * 3600 // time_reso) * city[6],
            np.ones(31 * 24 * 3600 // time_reso) * city[7],
            np.ones(30 * 24 * 3600 // time_reso) * city[8],
            np.ones(31 * 24 * 3600 // time_reso) * city[9],
            np.ones(30 * 24 * 3600 // time_reso) * city[10],
            np.ones(31 * 24 * 3600 // time_reso) * city[11],
        ]
    )

    # Check if building is in BUILDINGS, otherwise use building as building_file
    if building in BUILDINGS:
        # Get the absolute path of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the "Data" folder
        data_dir = os.path.join(script_dir, "Data/")
        building_file = data_dir + BUILDINGS[building][0]
        U_Wall = BUILDINGS[building][1]
    else:
        building_file = building

    # Check if weather is in WEATHER, otherwise use weather as weather_file
    if weather not in WEATHER:
        weather_file = [weather, groundtemp]
    else:
        weather_file = [data_dir + WEATHER[weather], groundtemp]

    # Update file paths if root is not user-defined
    if len(root) > 0:
        building_file = os.path.join(root, building_file)
        weather_file[0] = os.path.join(root, weather_file[0])

    # Get room information from the building file
    layers, n, all_zones = get_zones(building_file)
    print("###############All Zones from Ground############")
    for zone in all_zones:
        print(zone.name, " [Zone index]: ", zone.ind)
    print("###################################################")

    # Read the weather data and interpolate temperature values
    data = pvlib.iotools.read_epw(weather_file[0])
    oneyear = data[0]["temp_air"]
    num_datapoint = len(oneyear)
    x = np.arange(0, num_datapoint)
    y = np.array(oneyear)

    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, num_datapoint - 1, 1 / 3600 * time_reso)
    outtempdatanew = f(xnew)

    # Read the weather data and interpolate GHI values to the new time resolution
    oneyearrad = data[0]["ghi"]
    x = np.arange(0, num_datapoint)
    y = np.array(oneyearrad)

    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, num_datapoint - 1, 1 / 3600 * time_reso)
    solardatanew = f(xnew)

    # Define constants and calculate SHGC
    # TODO: where does SpecificHeat_avg come from?
    SpecificHeat_avg = 1000
    SHGC = shgc * shgc_weight * (max(data[0]["ghi"]) / (abs(data[1]["TZ"]) / 60))

    # Find neighboring rooms, resistance and capacitance tables, and window properties
    dicRoom, Rtable, Ctable, Windowtable = Nfind_neighbor(
        n, layers, U_Wall, SpecificHeat_avg
    )

    # Initialize connectivity matrix and ground connection list
    connectmap = np.zeros((n, n + 1))
    RCtable = Rtable / np.array([Ctable]).T
    ground_connectlist = np.zeros((n, 1))  # list to see which room connects to the ground
    groundrooms = layers[0]  # the first layer connects to the ground

    # Assign ground connection values and populate the connectivity matrix
    for room in groundrooms:
        ground_connectlist[room.ind] = (
            room.FloorArea * U_Wall[5] * ground_weight
        )  # for those rooms, assign 1/R table by floor area and u factor

    for i, zone in enumerate(all_zones):
        connect_list = dicRoom[zone.name]

        for number in connect_list:
            connectmap[i][number] = 1

    # Calculate occupancy, AC weight, weighted connection map, and non-linear term
    people_full = np.zeros((n, 1)) + np.array([full_occ]).T
    ACweight = np.diag(np.zeros(n) + AC_map) * max_power
    weightcmap = (
        np.concatenate(
            (
                people_full,
                ground_connectlist,
                np.zeros((n, 1)),
                ACweight,
                np.array([Windowtable * SHGC]).T,
            ),
            axis=-1,
        )
        / np.array([Ctable]).T
    )
    nonlinear = people_full / np.array([Ctable]).T

    # Store parameters in a dictionary for the simulation
    Parameter = {}
    Parameter["OutTemp"] = outtempdatanew
    Parameter["connectmap"] = connectmap
    Parameter["RCtable"] = RCtable
    Parameter["n"] = n
    Parameter["zonenames"] = [zone.name for zone in all_zones]
    Parameter["weightcmap"] = weightcmap
    Parameter["target"] = np.zeros(n) + target
    Parameter["gamma"] = reward_gamma
    Parameter["time_resolution"] = time_reso
    Parameter["ghi"] = (
        solardatanew
        / (abs(data[1]["TZ"]) / 60)
        / (max(data[0]["ghi"]) / (abs(data[1]["TZ"]) / 60))
    )
    Parameter["GroundTemp"] = weather_file[1]
    Parameter["Occupancy"] = activity_sch
    Parameter["ACmap"] = np.zeros(n) + AC_map
    Parameter["max_power"] = max_power
    Parameter["nonlinear"] = nonlinear
    Parameter["temp_range"] = temp_range
    Parameter["is_continuous_action"] = is_continuous_action

    return Parameter
