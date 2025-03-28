import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay

st.title("📊 Информация о тренировке модели")

# --- Пути к файлам с логами (можно сохранять в train_weather.py) ---
log_path = "models/train_log_finetuned.json"
cm_path = "models/confusion_finetuned.json"

# --- Кривая обучения ---
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        log = json.load(f)

    losses = log.get("valid_losses", [])
    timesec = log.get("training_time", None)

    st.subheader("📉 Кривая обучения (Loss по эпохам)")
    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses)+1), losses, marker="o")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

    if timesec:
        minutes = round(timesec / 60, 2)
        st.info(f"⏱️ Общее время обучения: {minutes} минут")
else:
    st.warning("Лог обучения не найден. Обновите train_weather.py для записи логов.")

# --- Распределение классов ---
st.subheader("📊 Распределение классов в датасете")
class_counts = {
    "001.Black_footed_Albatross": 60,
    "002.Laysan_Albatross": 60,
    "003.Sooty_Albatross": 58,
    "004.Groove_billed_Ani": 60,
    "005.Crested_Auklet": 44,
    "006.Least_Auklet": 41,
    "007.Parakeet_Auklet": 53,
    "008.Rhinoceros_Auklet": 48,
    "009.Brewer_Blackbird": 59,
    "010.Red_winged_Blackbird": 60,
    "011.Rusty_Blackbird": 60,
    "012.Yellow_headed_Blackbird": 56,
    "013.Bobolink": 60,
    "014.Indigo_Bunting": 60,
    "015.Lazuli_Bunting": 58,
    "016.Painted_Bunting": 58,
    "017.Cardinal": 57,
    "018.Spotted_Catbird": 45,
    "019.Gray_Catbird": 59,
    "020.Yellow_breasted_Chat": 59,
    "021.Eastern_Towhee": 60,
    "022.Chuck_will_Widow": 56,
    "023.Brandt_Cormorant": 59,
    "024.Red_faced_Cormorant": 52,
    "025.Pelagic_Cormorant": 60,
    "026.Bronzed_Cowbird": 60,
    "027.Shiny_Cowbird": 60,
    "028.Brown_Creeper": 59,
    "029.American_Crow": 60,
    "030.Fish_Crow": 60,
    "031.Black_billed_Cuckoo": 60,
    "032.Mangrove_Cuckoo": 53,
    "033.Yellow_billed_Cuckoo": 59,
    "034.Gray_crowned_Rosy_Finch": 59,
    "035.Purple_Finch": 60,
    "036.Northern_Flicker": 60,
    "037.Acadian_Flycatcher": 59,
    "038.Great_Crested_Flycatcher": 60,
    "039.Least_Flycatcher": 59,
    "040.Olive_sided_Flycatcher": 60,
    "041.Scissor_tailed_Flycatcher": 60,
    "042.Vermilion_Flycatcher": 60,
    "043.Yellow_bellied_Flycatcher": 59,
    "044.Frigatebird": 60,
    "045.Northern_Fulmar": 60,
    "046.Gadwall": 60,
    "047.American_Goldfinch": 60,
    "048.European_Goldfinch": 60,
    "049.Boat_tailed_Grackle": 60,
    "050.Eared_Grebe": 60,
    "051.Horned_Grebe": 60,
    "052.Pied_billed_Grebe": 60,
    "053.Western_Grebe": 60,
    "054.Blue_Grosbeak": 60,
    "055.Evening_Grosbeak": 60,
    "056.Pine_Grosbeak": 60,
    "057.Rose_breasted_Grosbeak": 60,
    "058.Pigeon_Guillemot": 58,
    "059.California_Gull": 60,
    "060.Glaucous_winged_Gull": 59,
    "061.Heermann_Gull": 60,
    "062.Herring_Gull": 60,
    "063.Ivory_Gull": 60,
    "064.Ring_billed_Gull": 60,
    "065.Slaty_backed_Gull": 50,
    "066.Western_Gull": 60,
    "067.Anna_Hummingbird": 60,
    "068.Ruby_throated_Hummingbird": 60,
    "069.Rufous_Hummingbird": 60,
    "070.Green_Violetear": 60,
    "071.Long_tailed_Jaeger": 60,
    "072.Pomarine_Jaeger": 60,
    "073.Blue_Jay": 60,
    "074.Florida_Jay": 60,
    "075.Green_Jay": 57,
    "076.Dark_eyed_Junco": 60,
    "077.Tropical_Kingbird": 60,
    "078.Gray_Kingbird": 59,
    "079.Belted_Kingfisher": 60,
    "080.Green_Kingfisher": 60,
    "081.Pied_Kingfisher": 60,
    "082.Ringed_Kingfisher": 60,
    "083.White_breasted_Kingfisher": 60,
    "084.Red_legged_Kittiwake": 53,
    "085.Horned_Lark": 60,
    "086.Pacific_Loon": 60,
    "087.Mallard": 60,
    "088.Western_Meadowlark": 60,
    "089.Hooded_Merganser": 60,
    "090.Red_breasted_Merganser": 60,
    "091.Mockingbird": 60,
    "092.Nighthawk": 60,
    "093.Clark_Nutcracker": 60,
    "094.White_breasted_Nuthatch": 60,
    "095.Baltimore_Oriole": 60,
    "096.Hooded_Oriole": 60,
    "097.Orchard_Oriole": 59,
    "098.Scott_Oriole": 60,
    "099.Ovenbird": 60,
    "100.Brown_Pelican": 60,
    "101.White_Pelican": 50,
    "102.Western_Wood_Pewee": 60,
    "103.Sayornis": 60,
    "104.American_Pipit": 60,
    "105.Whip_poor_Will": 49,
    "106.Horned_Puffin": 60,
    "107.Common_Raven": 59,
    "108.White_necked_Raven": 60,
    "109.American_Redstart": 60,
    "110.Geococcyx": 60,
    "111.Loggerhead_Shrike": 60,
    "112.Great_Grey_Shrike": 60,
    "113.Baird_Sparrow": 50,
    "114.Black_throated_Sparrow": 60,
    "115.Brewer_Sparrow": 59,
    "116.Chipping_Sparrow": 60,
    "117.Clay_colored_Sparrow": 59,
    "118.House_Sparrow": 60,
    "119.Field_Sparrow": 59,
    "120.Fox_Sparrow": 60,
    "121.Grasshopper_Sparrow": 60,
    "122.Harris_Sparrow": 60,
    "123.Henslow_Sparrow": 60,
    "124.Le_Conte_Sparrow": 59,
    "125.Lincoln_Sparrow": 59,
    "126.Nelson_Sharp_tailed_Sparrow": 59,
    "127.Savannah_Sparrow": 60,
    "128.Seaside_Sparrow": 60,
    "129.Song_Sparrow": 60,
    "130.Tree_Sparrow": 60,
    "131.Vesper_Sparrow": 60,
    "132.White_crowned_Sparrow": 60,
    "133.White_throated_Sparrow": 60,
    "134.Cape_Glossy_Starling": 60,
    "135.Bank_Swallow": 59,
    "136.Barn_Swallow": 60,
    "137.Cliff_Swallow": 60,
    "138.Tree_Swallow": 60,
    "139.Scarlet_Tanager": 60,
    "140.Summer_Tanager": 60,
    "141.Artic_Tern": 58,
    "142.Black_Tern": 60,
    "143.Caspian_Tern": 60,
    "144.Common_Tern": 60,
    "145.Elegant_Tern": 60,
    "146.Forsters_Tern": 60,
    "147.Least_Tern": 60,
    "148.Green_tailed_Towhee": 60,
    "149.Brown_Thrasher": 59,
    "150.Sage_Thrasher": 60,
    "151.Black_capped_Vireo": 51,
    "152.Blue_headed_Vireo": 60,
    "153.Philadelphia_Vireo": 59,
    "154.Red_eyed_Vireo": 60,
    "155.Warbling_Vireo": 60,
    "156.White_eyed_Vireo": 60,
    "157.Yellow_throated_Vireo": 59,
    "158.Bay_breasted_Warbler": 60,
    "159.Black_and_white_Warbler": 60,
    "160.Black_throated_Blue_Warbler": 59,
    "161.Blue_winged_Warbler": 60,
    "162.Canada_Warbler": 60,
    "163.Cape_May_Warbler": 60,
    "164.Cerulean_Warbler": 60,
    "165.Chestnut_sided_Warbler": 60,
    "166.Golden_winged_Warbler": 59,
    "167.Hooded_Warbler": 60,
    "168.Kentucky_Warbler": 59,
    "169.Magnolia_Warbler": 59,
    "170.Mourning_Warbler": 60,
    "171.Myrtle_Warbler": 60,
    "172.Nashville_Warbler": 60,
    "173.Orange_crowned_Warbler": 60,
    "174.Palm_Warbler": 60,
    "175.Pine_Warbler": 60,
    "176.Prairie_Warbler": 60,
    "177.Prothonotary_Warbler": 60,
    "178.Swainson_Warbler": 56,
    "179.Tennessee_Warbler": 59,
    "180.Wilson_Warbler": 60,
    "181.Worm_eating_Warbler": 59,
    "182.Yellow_Warbler": 60,
    "183.Northern_Waterthrush": 60,
    "184.Louisiana_Waterthrush": 60,
    "185.Bohemian_Waxwing": 60,
    "186.Cedar_Waxwing": 60,
    "187.American_Three_toed_Woodpecker": 50,
    "188.Pileated_Woodpecker": 60,
    "189.Red_bellied_Woodpecker": 60,
    "190.Red_cockaded_Woodpecker": 58,
    "191.Red_headed_Woodpecker": 60,
    "192.Downy_Woodpecker": 60,
    "193.Bewick_Wren": 60,
    "194.Cactus_Wren": 60,
    "195.Carolina_Wren": 60,
    "196.House_Wren": 59,
    "197.Marsh_Wren": 60,
    "198.Rock_Wren": 60,
    "199.Winter_Wren": 60,
    "200.Common_Yellowthroat": 60
}
df_counts = pd.DataFrame.from_dict(class_counts, orient='index', columns=["Количество"])
st.bar_chart(df_counts, horizontal=True)

# --- Confusion matrix ---
if os.path.exists(cm_path):
    with open(cm_path, "r") as f:
        cm_data = json.load(f)
        y_true = cm_data["true"]
        y_pred = cm_data["pred"]

    f1 = f1_score(y_true, y_pred, average='weighted')
    st.subheader("🎯 F1-score")
    st.write(f"**F1-score**: {f1:.3f}")

    # cm = confusion_matrix(y_true, y_pred)
    # fig2, ax2 = plt.subplots(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_counts.keys(), yticklabels=class_counts.keys())
    # st.pyplot(fig2)
else:
    st.warning("Файл с предсказаниями не найден. Добавьте сохранение в train_weather.py.")

print(df_counts)