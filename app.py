import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
import PyPDF2 
from pdfminer.layout import LAParams
from langchain.text_splitter import CharacterTextSplitter
import os
import openai
openai.api_base = "https://gpt35-curadoria.openai.azure.com/" 
openai.api_type = 'azure'
openai.api_version = "2023-05-15" 
openai.api_key = os.environ['AZURE_OPENAI_KEY'] 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pytesseract
from pdf2image import convert_from_path

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
from io import BytesIO
import tempfile
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader

from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from pathlib import Path
import io
from doctr.file_utils import is_tf_available

# if is_tf_available():
    # import tensorflow as tf

    # from backend.tensorflow import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    # if any(tf.config.experimental.list_physical_devices("gpu")):
        # forward_device = tf.device("/gpu:0")
    # else:
        # forward_device = tf.device("/cpu:0")

# else:
    # import torch

    # from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor

    # forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def convert_pdf_to_images_high_quality(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)

    enhanced_images = []
    for image in images:
        gray_image = image.convert('L')

        sharp_image = gray_image.filter(ImageFilter.SHARPEN)

        np_image = np.array(sharp_image)

        _, binary_image = cv2.threshold(np_image, 150, 255, cv2.THRESH_BINARY)

        denoised_image = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)

        final_image = Image.fromarray(denoised_image)

        enhanced_images.append(final_image)

    return enhanced_images


def extract_text_from_pdf(uploaded_files):

    def is_pdf_editable(reader):
        num_pages = len(reader.pages)
        for i in range(num_pages):
            page = reader.pages[i] 
            if page.extract_text().strip():
                return True
        return False

    def extract_text_with_ocr(uploaded_files):
        texts = {}
        model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

        for uploaded_file in uploaded_files:
            try:
                with Path(uploaded_file.name).open("wb") as f:
                    f.write(uploaded_file.getbuffer())

                jpg_images = []
                if uploaded_file.name.lower().endswith('.pdf'):
                    images = convert_pdf_to_images_high_quality(uploaded_file.name)
                    for index, image in enumerate(images):
                        temp_jpg_filename = f"{Path(uploaded_file.name).stem}_{index}.jpg"
                        image.save(temp_jpg_filename, format='JPEG')
                        jpg_images.append(temp_jpg_filename)
                else:
                    jpg_images = [uploaded_file.name]

                doc = DocumentFile.from_images(jpg_images)
                result = model(doc)
                extracted_text = ""
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                extracted_text += word.value + " "
                            extracted_text += "\n"

                texts[uploaded_file.name] = extracted_text

                for jpg_image in jpg_images:
                    Path(jpg_image).unlink(missing_ok=True)

            except Exception as e:
                texts[uploaded_file.name] = f"Erro ao extrair texto: {e}"
                for jpg_image in jpg_images:
                    Path(jpg_image).unlink(missing_ok=True)

            Path(uploaded_file.name).unlink(missing_ok=True)

        return texts

    extracted_texts = {}

    for uploaded_file in uploaded_files:
        try:
            reader = PyPDF2.PdfReader(uploaded_file, strict=False)

            if is_pdf_editable(reader):
                text = ""       
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]                           
                    text += page.extract_text()
                extracted_texts[uploaded_file.name] = text

            else:
                ocr_texts = extract_text_with_ocr([uploaded_file])
                extracted_texts[uploaded_file.name] = ocr_texts.get(uploaded_file.name, "")

        except Exception as e:
            extracted_texts[uploaded_file.name] = f"Erro ao extrair texto: {e}"

    return extracted_texts
   

def extract_text_with_ocr(uploaded_files):
    texts = {}
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

    for uploaded_file in uploaded_files:
        try:
            with Path(uploaded_file.name).open("wb") as f:
                f.write(uploaded_file.getbuffer())

            jpg_images = []
            if uploaded_file.name.lower().endswith('.pdf'):
                images = convert_pdf_to_images_high_quality(uploaded_file.name)
                for index, image in enumerate(images):
                    temp_jpg_filename = f"{Path(uploaded_file.name).stem}_{index}.jpg"
                    image.save(temp_jpg_filename, format='JPEG')
                    jpg_images.append(temp_jpg_filename)
            else:
                jpg_images = [uploaded_file.name]

            doc = DocumentFile.from_images(jpg_images)
            result = model(doc)
            extracted_text = ""
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            extracted_text += word.value + " "
                        extracted_text += "\n"

            texts[uploaded_file.name] = extracted_text

            for jpg_image in jpg_images:
                Path(jpg_image).unlink(missing_ok=True)

        except Exception as e:
            texts[uploaded_file.name] = f"Erro ao extrair texto: {e}"
            for jpg_image in jpg_images:
                Path(jpg_image).unlink(missing_ok=True)

        Path(uploaded_file.name).unlink(missing_ok=True)

    return texts


    
def extract_text_from_image(image):
    try:
        image = Image.open(image)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Erro ao extrair texto da imagem: {e}"    
    
def get_text_chunks(texts):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=400,
        length_function=len
    )
    chunks_dict = {}
    for filename, text in texts.items():
        chunks = text_splitter.split_text(text)
        chunks_dict[filename] = chunks
    return chunks_dict

def get_vectorstore(text_chunks_dict):
    embeddings = OpenAIEmbeddings(deployment="GPT_Embedding",deployment_id="GPT_Embedding",model="text-embedding-ada-002", chunk_size = 15)
    all_chunks = []
    for chunks in text_chunks_dict.values():
        all_chunks.extend(chunks)
    vectorstore = FAISS.from_texts(texts=all_chunks, embedding=embeddings)
    return vectorstore    

def get_conversation_chain(vectorstore):
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = "https://gpt35-curadoria.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = os.environ['AZURE_OPENAI_KEY']
    os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
    openai_api_key = os.environ['AZURE_OPENAI_KEY']
    llm = AzureChatOpenAI(deployment_name="GPT_4_Jornada_Cognitiva", 
                          openai_api_version="2023-07-01-preview", 
                          openai_api_key=openai_api_key, 
                          openai_api_base="https://gpt35-curadoria.openai.azure.com/")

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain
            
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    #st.write(response)
    
    st.session_state.user_questions.append(user_question)
    st.session_state.generated_responses.append(response['answer']) 

    for i in range(len(st.session_state.generated_responses)-1, -1, -1):
        message(st.session_state.generated_responses[i], key=f"generated_{i}", avatar_style='')
        message(st.session_state.user_questions[i], is_user=True, key=f"user_{i}", avatar_style='')

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat Jurídico", page_icon=None, layout="wide", initial_sidebar_state="expanded", menu_items=None)

    st.title("Sicoob Jurídico")    
    
    st.markdown("""
        <style>
            html, body, [class*="st-"] {
                font-size: calc(100% - 2pt);
            }
            /* Aumentar a fonte do campo de input */
            .stTextInput > div > div > input {
                font-size: calc(100% + 8pt);
            }
        </style>
    """, unsafe_allow_html=True)
    
    if "user_questions" not in st.session_state:
        st.session_state.user_questions = []
    if "generated_responses" not in st.session_state:
        st.session_state.generated_responses = []
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "pdf_texts" not in st.session_state:
        st.session_state.pdf_texts = None

    user_question = st.text_input("Pergunta sobre os autos:", key="user_input")
    submit_button = st.button("Enviar")
    
    if submit_button and user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Arquivos")
        
        uploaded_image = st.file_uploader("Carregue uma imagem", type=['png', 'jpg', 'jpeg'], key="image_uploader")
          
        if uploaded_image is not None:
            st.session_state.uploaded_image = uploaded_image
            st.image(uploaded_image, caption='Imagem Carregada', use_column_width=True)
            text_from_image = extract_text_from_image(uploaded_image)
            text_chunks_dict = get_text_chunks({"image_text": text_from_image})
            vectorstore = get_vectorstore(text_chunks_dict)
            st.session_state.conversation = get_conversation_chain(vectorstore)
        elif st.session_state.uploaded_image and st.button("Remover Imagem"):
            st.session_state.uploaded_image = None
            if st.session_state.pdf_texts:
                text_chunks_dict = get_text_chunks(st.session_state.pdf_texts)
                vectorstore = get_vectorstore(text_chunks_dict)
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
        pdf_docs = st.file_uploader("Carregue os arquivos PDF e clique em 'Processar'", accept_multiple_files=True)
        if st.button("Processar PDFs"):
            with st.spinner("Processando"):
                st.session_state.pdf_texts = extract_text_from_pdf(pdf_docs)
                text_chunks_dict = get_text_chunks(st.session_state.pdf_texts)
                vectorstore = get_vectorstore(text_chunks_dict)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Pronto para conversar usando o documento!")

if __name__ == '__main__':
    main()


