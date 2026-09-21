from datetime import datetime
from pathlib import Path
from uuid import uuid4
import sqlite3

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------
# Grundeinstellungen
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fundbüro Kleidung",
    page_icon="👕",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
DATABASE = DATA_DIR / "fundbuero.db"

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

CATEGORIES = {
    "T-Shirt": "a photo of a t-shirt",
    "Hemd": "a photo of a shirt",
    "Pullover": "a photo of a sweater or pullover",
    "Hoodie": "a photo of a hoodie",
    "Jacke": "a photo of a jacket",
    "Mantel": "a photo of a coat",
    "Hose": "a photo of trousers or pants",
    "Jeans": "a photo of jeans",
    "Rock": "a photo of a skirt",
    "Kleid": "a photo of a dress",
    "Shorts": "a photo of shorts",
    "Schuhe": "a photo of shoes",
    "Mütze": "a photo of a hat or beanie",
    "Schal": "a photo of a scarf",
    "Handschuhe": "a photo of gloves",
    "Sonstiges": "a photo of another clothing item",
}


# ---------------------------------------------------------
# Datenbank
# ---------------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            category TEXT NOT NULL,
            upload_date TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def add_entry(filename, image_path, category):
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO entries
        (filename, image_path, category, upload_date)
        VALUES (?, ?, ?, ?)
        """,
        (
            filename,
            str(image_path),
            category,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
    )

    connection.commit()
    connection.close()


def get_entries(search_text=""):
    connection = get_connection()

    if search_text.strip():
        rows = connection.execute(
            """
            SELECT *
            FROM entries
            WHERE category LIKE ?
               OR filename LIKE ?
            ORDER BY id DESC
            """,
            (
                f"%{search_text}%",
                f"%{search_text}%",
            ),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT *
            FROM entries
            ORDER BY id DESC
            """
        ).fetchall()

    connection.close()
    return rows


initialize_database()


# ---------------------------------------------------------
# Hugging-Face-Modell laden
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    model_name = "patrickjohncyh/fashion-clip"

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    model.eval()

    return processor, model


def classify_image(image):
    processor, model = load_model()

    category_names = list(CATEGORIES.keys())
    category_prompts = list(CATEGORIES.values())

    inputs = processor(
        text=category_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = outputs.logits_per_image.softmax(dim=1)[0]
    best_index = int(torch.argmax(probabilities))

    return (
        category_names[best_index],
        float(probabilities[best_index]),
    )


# ---------------------------------------------------------
# Hilfsfunktionen für die Darstellung
# ---------------------------------------------------------

def show_entry(entry):
    columns = st.columns([1, 2])

    image_path = Path(entry["image_path"])

    with columns[0]:
        if image_path.exists():
            st.image(
                str(image_path),
                use_container_width=True,
            )
        else:
            st.warning("Bild nicht gefunden.")

    with columns[1]:
        st.subheader(entry["category"])
        st.write(f"**Hochgeladen:** {entry['upload_date']}")
        st.caption(f"Dateiname: {entry['filename']}")


def show_entry_grid(entries):
    if not entries:
        st.info("Noch keine Einträge vorhanden.")
        return

    for index in range(0, len(entries), 3):
        columns = st.columns(3)

        for column, entry in zip(columns, entries[index:index + 3]):
            image_path = Path(entry["image_path"])

            with column:
                if image_path.exists():
                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )

                st.markdown(f"### {entry['category']}")
                st.caption(entry["upload_date"])
                st.divider()


# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Startseite"

st.sidebar.title("👕 Fundbüro")

page = st.sidebar.radio(
    "Navigation",
    [
        "Startseite",
        "Kleidungsstück hochladen",
        "Alle Einträge",
    ],
    index=[
        "Startseite",
        "Kleidungsstück hochladen",
        "Alle Einträge",
    ].index(st.session_state["page"]),
)

st.session_state["page"] = page



# ---------------------------------------------------------
# Startseite
# ---------------------------------------------------------

if page == "Startseite":
    st.title("👕 Fundbüro für Kleidungsstücke")
    st.write(
        "Lade ein Foto eines gefundenen Kleidungsstücks hoch. "
        "Die App erkennt automatisch die passende Kategorie."
    )

    st.divider()

    search_text = st.text_input(
        "🔎 Einträge durchsuchen",
        placeholder="Zum Beispiel: Jacke, Hose oder Schuhe",
    )

    st.subheader("Neueste Einträge")

    entries = get_entries(search_text)
    show_entry_grid(entries[:6])

    st.divider()

    if st.button("Alle Einträge anzeigen", type="primary"):
        st.session_state["page"] = "Alle Einträge"
        st.rerun()


# ---------------------------------------------------------
# Upload-Seite
# ---------------------------------------------------------

elif page == "Kleidungsstück hochladen":
    st.title("Kleidungsstück hochladen")

    uploaded_file = st.file_uploader(
        "Foto auswählen",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Ausgewähltes Bild",
            width=400,
        )

        if st.button("Bild analysieren", type="primary"):
            with st.spinner(
                "Das Kleidungsstück wird analysiert. "
                "Beim ersten Start kann das Laden des Modells etwas dauern."
            ):
                category, confidence = classify_image(image)

            st.session_state["classification"] = category
            st.session_state["confidence"] = confidence

        if "classification" in st.session_state:
            category = st.session_state["classification"]
            confidence = st.session_state["confidence"]

            st.success(
                f"Erkannte Kategorie: {category} "
                f"({confidence * 100:.1f} % Modellvertrauen)"
            )

            corrected_category = st.selectbox(
                "Kategorie bestätigen oder korrigieren",
                list(CATEGORIES.keys()),
                index=list(CATEGORIES.keys()).index(category),
            )

            if st.button("Eintrag speichern", type="primary"):
                file_extension = Path(uploaded_file.name).suffix.lower()
                unique_filename = f"{uuid4().hex}{file_extension}"
                image_path = IMAGE_DIR / unique_filename

                image.save(image_path)

                add_entry(
                    filename=uploaded_file.name,
                    image_path=image_path,
                    category=corrected_category,
                )

                st.success("Der Eintrag wurde erfolgreich gespeichert.")

                st.session_state.pop("classification", None)
                st.session_state.pop("confidence", None)

                st.balloons()


# ---------------------------------------------------------
# Alle Einträge
# ---------------------------------------------------------

elif page == "Alle Einträge":
    st.title("Alle Einträge")

    search_text = st.text_input(
        "🔎 Suche",
        placeholder="Kategorie oder Dateiname",
    )

    entries = get_entries(search_text)

    st.write(f"{len(entries)} Einträge gefunden.")

    show_entry_grid(entries)

    if entries:
        dataframe = pd.DataFrame(
            [
                {
                    "Kategorie": entry["category"],
                    "Hochgeladen": entry["upload_date"],
                    "Dateiname": entry["filename"],
                }
                for entry in entries
            ]
        )

        with st.expander("Tabellarische Ansicht"):
            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True,
            )
