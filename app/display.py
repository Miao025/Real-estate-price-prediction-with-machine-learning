import os
import sys
import streamlit as st
import pandas as pd
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
model_path = os.path.join(current_dir, 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title('House Price Prediction')
st.header('Enter House Features:')

type = st.selectbox('🏘️Property Type (required)', ['APARTMENT', 'HOUSE'])
subtype = st.selectbox('🏠Property Subtype (required)', ['APARTMENT', 'HOUSE', 'FLAT_STUDIO', 'DUPLEX', 'PENTHOUSE', 'GROUND_FLOOR', 'APARTMENT_BLOCK', 'MANSION', 'EXCEPTIONAL_PROPERTY', 'MIXED_USE_BUILDING', 'TRIPLEX', 'LOFT', 'VILLA', 'TOWN_HOUSE', 'CHALET', 'MANOR_HOUSE', 'SERVICE_FLAT', 'KOT', 'FARMHOUSE', 'BUNGALOW', 'COUNTRY_COTTAGE', 'OTHER_PROPERTY', 'CASTLE', 'PAVILION'])
bedroom = st.text_input('🛏️Number of Bedrooms')
bedroomCount = float(bedroom) if bedroom else None
bathroom = st.text_input('🛀Number of Bathrooms')
bathroomCount = float(bathroom) if bathroom else None
region = st.selectbox('🚩Region (required)', ['Brussels', 'Walloon', 'Flanders'])
province = st.selectbox('🚩Province (required)', ['Brussels', 'Luxembourg', 'Antwerp', 'West Flanders', 'Liège', 'East Flanders', 'Walloon Brabant', 'Limburg', 'Flemish Brabant', 'Namur', 'Hainaut'])
post = st.text_input('📫Post Code (required)')
postCode = int(post) if post else None
habitable = st.text_input('📏Habitable Surface(m²)')
habitableSurface = float(habitable) if habitable else None
land = st.text_input('📏Land Surface(m²)')
landSurface = float(land) if land else None
attic = st.radio('🚪Has Attic?', ['Yes', 'No', 'Not specified'])
hasAttic = True if attic == 'Yes' else None
basement = st.radio('🚪Has Basement?', ['Yes', 'No', 'Not specified'])
hasBasement = True if basement == 'Yes' else None
dressing = st.radio('👗Has Dressing Room?', ['Yes', 'No', 'Not specified'])
hasDressingRoom = True if dressing == 'Yes' else None
dining = st.radio('🍽️Has Dining Room?', ['Yes', 'No', 'Not specified'])
hasDiningRoom = True if dining == 'Yes' else None
diningS = st.text_input('🍽️Dining Room Surface(m²)')
diningRoomSurface = float(diningS) if diningS else None
building = st.selectbox('🏠Building Condition',['GOOD', 'TO_BE_DONE_UP', 'AS_NEW', 'JUST_RENOVATED', 'TO_RENOVATE', 'TO_RESTORE','NOT_SPECIFIED'])
buildingCondition = None if building=='NOT_SPECIFIED' else building
construction = st.text_input('🏠Building Construction Year')
buildingConstructionYear = float(construction) if construction else None
facede = st.text_input('🏠Number of facedes')
facedeCount = float(facede) if facede else None
floor = st.text_input('🏠Number of floors')
floorCount = float(floor) if floor else None
street = st.text_input('🏠Street Facede Width (m)')
streetFacadeWidth = float(street) if street else None
lift = st.radio('🛗Has Lift', ['Yes', 'No', 'Not specified'])
hasLift = True if lift == 'Yes' else None
flood = st.selectbox('💧Flood Zone Type',['NON_FLOOD_ZONE', 'RECOGNIZED_FLOOD_ZONE', 'POSSIBLE_FLOOD_ZONE', 'CIRCUMSCRIBED_WATERSIDE_ZONE', 'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE', 'RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE', 'CIRCUMSCRIBED_FLOOD_ZONE', 'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE', 'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE','NOT_SPECIFIED'])
floodZoneType = None if flood=='NOT_SPECIFIED' else flood
heat = st.radio('🔥Has Heat Pump', ['Yes', 'No', 'Not specified'])
hasHeatPump = True if heat == 'Yes' else None
heatT = st.selectbox('🔥Heating Type',['GAS', 'FUELOIL', 'ELECTRIC', 'PELLET', 'SOLAR', 'CARBON', 'WOOD', 'NOT_SPECIFIED'])
heatingType = None if heatT=='NOT_SPECIFIED' else heatT
photo = st.radio('☀️Has Photovoltaic Panels', ['Yes', 'No', 'Not specified'])
hasPhotovoltaicPanels = True if photo == 'Yes' else None
therm = st.radio('🌡️Has Thermic Panels', ['Yes', 'No', 'Not specified'])
hasThermicPanels = True if therm == 'Yes' else None
kitchenS = st.text_input('🧑‍🍳Kitchen Surface(m²)')
kitchenSurface = float(kitchenS) if kitchenS else None
kitchenT = st.selectbox('🧑‍🍳Kitchen Type',['SEMI_EQUIPPED', 'INSTALLED', 'HYPER_EQUIPPED', 'NOT_INSTALLED', 'USA_UNINSTALLED', 'USA_HYPER_EQUIPPED', 'USA_INSTALLED', 'USA_SEMI_EQUIPPED', 'NOT_SPECIFIED'])
kitchenType = None if kitchenT=='NOT_SPECIFIED' else kitchenT
living = st.radio('🛋️Has Living Room', ['Yes', 'No', 'Not specified'])
hasLivingRoom = True if living == 'Yes' else None
livingS = st.text_input('🛋️Living Room Surface(m²)')
livingRoomSurface = float(livingS) if livingS else None
garden = st.radio('🌺Has Garden', ['Yes', 'No', 'Not specified'])
hasGarden = True if garden == 'Yes' else None
gardenS = st.text_input('🌺Garden Surface(m²)')
gardenSurface = float(gardenS) if gardenS else None
gardenO = st.selectbox('🌺Garden Type',['NORTH_WEST', 'NORTH', 'NORTH_EAST', 'EAST', 'SOUTH_EAST', 'SOUTH_WEST', 'SOUTH', 'WEST', 'NOT_SPECIFIED'])
gardenOrientation = None if gardenO=='NOT_SPECIFIED' else gardenO
parkI = st.text_input('🚗Number of Indoor Parking')
parkingCountIndoor = float(parkI) if parkI else None
parkO = st.text_input('🚗Number of Outdoor Parking')
parkingCountOutdoor = float(parkO) if parkO else None
air = st.radio('🍃Has Air Conditioning', ['Yes', 'No', 'Not specified'])
hasAirConditioning = True if air == 'Yes' else None
arm = st.radio('🚪Has Armored Door', ['Yes', 'No', 'Not specified'])
hasArmoredDoor = True if arm == 'Yes' else None
vision = st.radio('📞Has Vision Phone', ['Yes', 'No', 'Not specified'])
hasVisiophone = True if vision == 'Yes' else None
office = st.radio('📂Has Office', ['Yes', 'No', 'Not specified'])
hasOffice = True if office == 'Yes' else None
toilet = st.text_input('🚽Number of Toilets')
toiletCount = float(toilet) if toilet else None
swim = st.radio('🏊Has Swimming Pool', ['Yes', 'No', 'Not specified'])
hasSwimmingPool = True if swim == 'Yes' else None
fire = st.radio('🔥Has Fire Place', ['Yes', 'No', 'Not specified'])
hasFireplace = True if fire == 'Yes' else None
terrace = st.radio('🏡Has Terrace', ['Yes', 'No', 'Not specified'])
hasTerrace = True if terrace == 'Yes' else None
terraceS = st.text_input('🏡Terrace Surface(m²)')
terraceSurface = float(terraceS) if terraceS else None
terraceO = st.selectbox('🏡Terrace Orientation', ['SOUTH_WEST', 'EAST', 'WEST', 'SOUTH_EAST', 'SOUTH', 'NORTH_WEST', 'NORTH', 'NORTH_EAST', 'NOT_SPECIFIED'])
terraceOrientation = None if terraceO=='NOT_SPECIFIED' else terraceO
epc = st.selectbox('EPC Score', ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'NOT_SPECIFIED'])
epcScore = None if epc=='NOT_SPECIFIED' else epc

if not postCode:
    st.error('Please enter the required fields.')

submit = st.button('Predictive Price', disabled=not postCode)

if submit:
    user_input = pd.DataFrame([{
        'type': type,
        'subtype': subtype,
        'bedroomCount': bedroomCount,
        'bathroomCount': bathroomCount,
        'province': province,
        'postCode': postCode,
        'habitableSurface': habitableSurface,
        'hasAttic': hasAttic,
        'hasBasement': hasBasement,
        'hasDressingRoom': hasDressingRoom,
        'diningRoomSurface': diningRoomSurface,
        'hasDiningRoom': hasDiningRoom,
        'buildingCondition': buildingCondition,
        'buildingConstructionYear': buildingConstructionYear,
        'facedeCount': facedeCount,
        'floorCount': floorCount,
        'streetFacadeWidth': streetFacadeWidth,
        'hasLift': hasLift,
        'floodZoneType': floodZoneType,
        'heatingType': heatingType,
        'hasHeatPump': hasHeatPump,
        'hasPhotovoltaicPanels': hasPhotovoltaicPanels,
        'hasThermicPanels': hasThermicPanels,
        'kitchenSurface': kitchenSurface,
        'kitchenType': kitchenType,
        'landSurface': landSurface,
        'hasLivingRoom': hasLivingRoom,
        'livingRoomSurface': livingRoomSurface,
        'hasGarden': hasGarden,
        'gardenSurface': gardenSurface,
        'gardenOrientation': gardenOrientation,
        'parkingCountIndoor': parkingCountIndoor,
        'parkingCountOutdoor': parkingCountOutdoor,
        'hasAirConditioning': hasAirConditioning,
        'hasArmoredDoor': hasArmoredDoor,
        'hasVisiophone': hasVisiophone,
        'hasOffice': hasOffice,
        'toiletCount': toiletCount,
        'hasSwimmingPool': hasSwimmingPool,
        'hasFireplace': hasFireplace,
        'hasTerrace': hasTerrace,
        'terraceSurface': terraceSurface,
        'terraceOrientation': terraceOrientation,
        'epcScore': epcScore,
        'region': region
    }])
    prediction = model.predict(user_input)
    st.success(f'Estimated price: €{prediction[0]:,.0f}')