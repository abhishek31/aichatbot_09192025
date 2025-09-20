
import os
import io
import time
import glob
import json
import base64
import re
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import PIL
import fitz
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
    Content,
    Part
)
from vertexai.vision_models import Image as vision_model_Image
import streamlit as st
from google.cloud import speech
import wave


# Functions for getting text and image embeddings

def get_text_embedding_from_text_embedding_model(
    text: str,
    text_embedding_model,
    return_array: Optional[bool] = False
) -> list:
    """
    Generates a numerical text embedding from a provided text input using a text embedding model.

    Args:
        text: The input text string to be embedded.
        return_array: If True, returns the embedding as a NumPy array.
                      If False, returns the embedding as a list. (Default: False)

    Returns:
        list or numpy.ndarray: A 768-dimensional vector representation of the input text.
                               The format (list or NumPy array) depends on the
                               value of the 'return_array' parameter.
    """
    embeddings = text_embedding_model.get_embeddings([text])
    text_embedding = [embedding.values for embedding in embeddings][0]

    if return_array:
        text_embedding = np.fromiter(text_embedding, dtype=float)

    # returns 768 dimensional array
    return text_embedding


def get_image_embedding_from_multimodal_embedding_model(
    image_uri: str,
    multimodal_embedding_model,
    embedding_size: int = 512,
    text: Optional[str] = None,
    return_array: Optional[bool] = False,
) -> list:
    """Extracts an image embedding from a multimodal embedding model.
    The function can optionally utilize contextual text to refine the embedding.

    Args:
        image_uri (str): The URI (Uniform Resource Identifier) of the image to process.
        text (Optional[str]): Optional contextual text to guide the embedding generation. Defaults to "".
        embedding_size (int): The desired dimensionality of the output embedding. Defaults to 512.
        return_array (Optional[bool]): If True, returns the embedding as a NumPy array.
        Otherwise, returns a list. Defaults to False.

    Returns:
        list: A list containing the image embedding values. If `return_array` is True, returns a NumPy array instead.
    """
    # image = Image.load_from_file(image_uri)
    print(f"92 {image_uri}")
    image = vision_model_Image.load_from_file(image_uri)
    embeddings = multimodal_embedding_model.get_embeddings(
        image=image, contextual_text=text, dimension=embedding_size
    )  # 128, 256, 512, 1408
    image_embedding = embeddings.image_embedding

    if return_array:
        image_embedding = np.fromiter(image_embedding, dtype=float)

    return image_embedding


def get_pdf_doc_object(pdf_path: str) -> tuple[fitz.Document, int]:
    """
    Opens a PDF file using fitz.open() and returns the PDF document object and the number of pages.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A tuple containing the `fitz.Document` object and the number of pages in the PDF.

    Raises:
        FileNotFoundError: If the provided PDF path is invalid.

    """

    # Open the PDF file
    doc: fitz.Document = fitz.open(pdf_path)

    # Get the number of pages in the PDF file
    num_pages: int = len(doc)

    return doc, num_pages


# Add colors to the print
class Color:
    """
    This class defines a set of color codes that can be used to print text in different colors.
    This will be used later to print citations and results to make outputs more readable.
    """

    PURPLE: str = "\033[95m"
    CYAN: str = "\033[96m"
    DARKCYAN: str = "\033[36m"
    BLUE: str = "\033[94m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"
    RED: str = "\033[91m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"
    END: str = "\033[0m"


def get_text_overlapping_chunk(
    text: str, character_limit: int = 1000, overlap: int = 100
) -> dict:
    """
    * Breaks a text document into chunks of a specified size, with an overlap between chunks to preserve context.
    * Takes a text document, character limit per chunk, and overlap between chunks as input.
    * Returns a dictionary where the keys are chunk numbers and the values are the corresponding text chunks.

    Args:
        text: The text document to be chunked.
        character_limit: Maximum characters per chunk (defaults to 1000).
        overlap: Number of overlapping characters between chunks (defaults to 100).

    Returns:
        A dictionary where keys are chunk numbers and values are the corresponding text chunks.

    Raises:
        ValueError: If `overlap` is greater than `character_limit`.

    """

    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Initialize variables
    chunk_number = 1
    chunked_text_dict = {}

    # Iterate over text with the given limit and overlap
    for i in range(0, len(text), character_limit - overlap):
        end_index = min(i + character_limit, len(text))
        chunk = text[i:end_index]

        # Encode and decode for consistent encoding
        chunked_text_dict[chunk_number] = chunk.encode("ascii", "ignore").decode(
            "utf-8", "ignore"
        )

        # Increment chunk number
        chunk_number += 1

    return chunked_text_dict


def get_page_text_embedding(text_data: Union[dict, str], text_embedding_model) -> dict:
    """
    * Generates embeddings for each text chunk using a specified embedding model.
    * Takes a dictionary of text chunks and an embedding size as input.
    * Returns a dictionary where the keys are chunk numbers and the values are the corresponding embeddings.

    Args:
        text_data: Either a dictionary of pre-chunked text or the entire page text.
        embedding_size: Size of the embedding vector (defaults to 128).

    Returns:
        A dictionary where keys are chunk numbers or "text_embedding" and values are the corresponding embeddings.

    """

    embeddings_dict = {}

    if not text_data:
        return embeddings_dict

    if isinstance(text_data, dict):
        # Process each chunk
        # print(text_data)
        for chunk_number, chunk_value in text_data.items():
            text_embd = get_text_embedding_from_text_embedding_model(text=chunk_value, text_embedding_model=text_embedding_model)
            embeddings_dict[chunk_number] = text_embd
    else:
        # Process the first 1000 characters of the page text
        text_embd = get_text_embedding_from_text_embedding_model(text=text_data, text_embedding_model=text_embedding_model)
        embeddings_dict["text_embedding"] = text_embd

    return embeddings_dict


def get_chunk_text_metadata(
    page: fitz.Page,
    text_embedding_model,
    character_limit: int = 1000,
    overlap: int = 100,
    embedding_size: int = 128,
) -> tuple[str, dict, dict, dict]:
    """
    * Extracts text from a given page object, chunks it, and generates embeddings for each chunk.
    * Takes a page object, character limit per chunk, overlap between chunks, and embedding size as input.
    * Returns the extracted text, the chunked text dictionary, and the chunk embeddings dictionary.

    Args:
        page: The fitz.Page object to process.
        character_limit: Maximum characters per chunk (defaults to 1000).
        overlap: Number of overlapping characters between chunks (defaults to 100).
        embedding_size: Size of the embedding vector (defaults to 128).

    Returns:
        A tuple containing:
            - Extracted page text as a string.
            - Dictionary of embeddings for the entire page text (key="text_embedding").
            - Dictionary of chunked text (key=chunk number, value=text chunk).
            - Dictionary of embeddings for each chunk (key=chunk number, value=embedding).

    Raises:
        ValueError: If `overlap` is greater than `character_limit`.

    """

    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Extract text from the page
    text: str = page.get_text().encode("ascii", "ignore").decode("utf-8", "ignore")

    # Get whole-page text embeddings
    page_text_embeddings_dict: dict = get_page_text_embedding(text, text_embedding_model)

    # Chunk the text with the given limit and overlap
    chunked_text_dict: dict = get_text_overlapping_chunk(text, character_limit, overlap)
    # print(chunked_text_dict)

    # Get embeddings for the chunks
    chunk_embeddings_dict: dict = get_page_text_embedding(chunked_text_dict, text_embedding_model)
    # print(chunk_embeddings_dict)

    # Return all extracted data
    return text, page_text_embeddings_dict, chunked_text_dict, chunk_embeddings_dict


def get_image_for_gemini(
    doc: fitz.Document,
    image: tuple,
    image_no: int,
    image_save_dir: str,
    file_name: str,
    page_num: int,
) -> Tuple[Image, str]:
    """
    Extracts an image from a PDF document, converts it to JPEG format (handling color conversions), saves it, and loads it as a PIL Image Object.
    """

    xref = image[0]
    pix = fitz.Pixmap(doc, xref)

    # Check and convert color space if needed
    if pix.colorspace not in (fitz.csGRAY, fitz.csRGB, fitz.csCMYK):
        pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB, which JPEG supports

    # Now save as JPEG (no need for pix.tobytes("jpeg"))
    image_name = f"{file_name}_image_{page_num+1}_{image_no}_{xref}.jpeg"
    image_name = os.path.join(image_save_dir, image_name)
    os.makedirs(image_save_dir, exist_ok=True)
    pix.save(image_name)

    image_for_gemini = Image.load_from_file(image_name)
    return image_for_gemini, image_name



def get_gemini_response(
    generative_multimodal_model,
    model_input: List[str],
    stream: bool = True,
    temperature=0.2,
    max_output_tokens=2048,
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    print_exception: bool = False,
) -> str:
    """
    This function generates text in response to a list of model inputs.

    Args:
        model_input: A list of strings representing the inputs to the model.
        stream: Whether to generate the response in a streaming fashion (returning chunks of text at a time) or all at once. Defaults to False.

    Returns:
        The generated text as a string.
    """
    config = GenerationConfig(
        temperature=temperature, max_output_tokens=max_output_tokens
    )
    response = generative_multimodal_model.generate_content(
        model_input,
        generation_config=config,
        stream=stream,
        safety_settings=safety_settings,
    )
    response_list = []

    for chunk in response:
        try:
            response_list.append(chunk.text)
        except Exception as e:
            if print_exception:
              print(
                  "Exception occurred while calling gemini. Something is blocked. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----",
                  e,
              )
            else:
              print("Exception occurred while calling gemini. Something is blocked. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----")
            response_list.append("**Something blocked.**")
            continue
    response = "".join(response_list)

    return response

def get_gemini_chat_response(
    generative_multimodal_model,
    model_input: List[str],
    stream: bool = True,
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    print_exception: bool = False,
    llm_temperature = 1,
    llm_max_output_tokens = 2048

) -> str:
    """
    This function generates text in response to a list of model inputs.

    Args:
        model_input: A list of strings representing the inputs to the model.
        stream: Whether to generate the response in a streaming fashion (returning chunks of text at a time) or all at once. Defaults to False.

    Returns:
        The generated text as a string.
    """
    config = GenerationConfig(
        temperature=llm_temperature, max_output_tokens=llm_max_output_tokens
    )
    response = generative_multimodal_model.send_message(
        model_input,
        generation_config=config,
        stream=stream,
        safety_settings=safety_settings
    )

    response_list = []

    for chunk in response:
        try:
            response_list.append(chunk.text)
        except Exception as e:
            if print_exception:
              print(
                  "Exception occurred while calling gemini. Something is blocked. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----",
                  e,
              )
            else:
              print("Exception occurred while calling gemini. Something is blocked. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----")
            response_list.append("**Something blocked.**")
            continue
    response = "".join(response_list)

    return response


def get_text_metadata_df(
    filename: str, text_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    This function takes a filename and a text metadata dictionary as input,
    iterates over the text metadata dictionary and extracts the text, chunk text,
    and chunk embeddings for each page, creates a Pandas DataFrame with the
    extracted data, and returns it.

    Args:
        filename: The filename of the document.
        text_metadata: A dictionary containing the text metadata for each page.

    Returns:
        A Pandas DataFrame with the extracted text, chunk text, and chunk embeddings for each page.
    """

    final_data_text: List[Dict] = []

    for key, values in text_metadata.items():
        for chunk_number, chunk_text in values["chunked_text_dict"].items():
            data: Dict = {}
            data["file_name"] = filename
            data["page_num"] = int(key) + 1
            data["text"] = values["text"]
            data["text_embedding_page"] = values["page_text_embeddings"][
                "text_embedding"
            ]
            data["chunk_number"] = chunk_number
            data["chunk_text"] = chunk_text
            data["text_embedding_chunk"] = values["chunk_embeddings_dict"][chunk_number]

            final_data_text.append(data)

    return_df = pd.DataFrame(final_data_text)
    return_df = return_df.reset_index(drop=True)
    return return_df


def get_image_metadata_df(
    filename: str, image_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    This function takes a filename and an image metadata dictionary as input,
    iterates over the image metadata dictionary and extracts the image path,
    image description, and image embeddings for each image, creates a Pandas
    DataFrame with the extracted data, and returns it.

    Args:
        filename: The filename of the document.
        image_metadata: A dictionary containing the image metadata for each page.

    Returns:
        A Pandas DataFrame with the extracted image path, image description, and image embeddings for each image.
    """

    final_data_image: List[Dict] = []
    for key, values in image_metadata.items():
        for _, image_values in values.items():
            data: Dict = {}
            data["file_name"] = filename
            data["page_num"] = int(key) + 1
            data["img_num"] = int(image_values["img_num"])
            data["img_path"] = image_values["img_path"]
            data["img_desc"] = image_values["img_desc"]
            # data["mm_embedding_from_text_desc_and_img"] = image_values[
            #     "mm_embedding_from_text_desc_and_img"
            # ]
            data["mm_embedding_from_img_only"] = image_values[
                "mm_embedding_from_img_only"
            ]
            data["text_embedding_from_image_description"] = image_values[
                "text_embedding_from_image_description"
            ]
            final_data_image.append(data)

    return_df = pd.DataFrame(final_data_image).dropna()
    return_df = return_df.reset_index(drop=True)
    return return_df


def get_document_metadata(
    generative_multimodal_model,
    pdf_paths: str,
    image_save_dir: str,
    image_description_prompt: str,
    text_embedding_model,
    multimodal_embedding_model,
    embedding_size: int = 128,
    image_text_extraction_max_output_tokens=2048,
    image_text_extraction_temperature=0.2,
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    add_sleep_after_page: bool = False,
    sleep_time_after_page: int = 2,
    add_sleep_after_document: bool = False,
    sleep_time_after_document: int = 2,
    none_chk = False,
    pages_changed= None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a PDF path, an image save directory, an image description prompt, an embedding size, and a text embedding text limit as input.

    Args:
        pdf_path: The path to the PDF document.
        image_save_dir: The directory where extracted images should be saved.
        image_description_prompt: A prompt to guide Gemini for generating image descriptions.
        embedding_size: The dimensionality of the embedding vectors.
        text_emb_text_limit: The maximum number of tokens for text embedding.

    Returns:
        A tuple containing two DataFrames:
            * One DataFrame containing the extracted text metadata for each page of the PDF, including the page text, chunked text dictionaries, and chunk embedding dictionaries.
            * Another DataFrame containing the extracted image metadata for each image in the PDF, including the image path, image description, image embeddings (with and without context), and image description text embedding.
    """
    text_metadata_df_final, image_metadata_df_final = pd.DataFrame(), pd.DataFrame()
    for pdf_path in pdf_paths:
        print(
            "\n\n",
            "Processing the file: ---------------------------------",
            pdf_path,
            "\n\n",
        )

        doc, num_pages = get_pdf_doc_object(pdf_path)
        file_name = os.path.split(pdf_path)[-1]

        text_metadata: Dict[Union[int, str], Dict] = {}
        image_metadata: Dict[Union[int, str], Dict] = {}
        progress_bar = st.progress(0)
        for page_num in range(num_pages):
            if pages_changed is not None:
                if page_num + 1 not in pages_changed:
                    print(f"Skipping page: {page_num + 1}")
                    progress_bar.progress((page_num + 1) / num_pages, text = f"Skipping page {page_num + 1} of {num_pages}")
                    continue
            print(f"Processing page: {page_num + 1}")
            if none_chk and page_num+1 < num_pages:
                print("breaking!")
                break
            progress_bar.progress((page_num + 1) / num_pages, text = f"Processing page {page_num + 1} of {num_pages}")

            page = doc[page_num]

            text = page.get_text()
            (
                text,
                page_text_embeddings_dict,
                chunked_text_dict,
                chunk_embeddings_dict,
            ) = get_chunk_text_metadata(page, text_embedding_model= text_embedding_model, embedding_size=embedding_size)

            text_metadata[page_num] = {
                "text": text,
                "page_text_embeddings": page_text_embeddings_dict,
                "chunked_text_dict": chunked_text_dict,
                "chunk_embeddings_dict": chunk_embeddings_dict,
            }

            images = page.get_images()
            image_metadata[page_num] = {}

            for image_no, image in enumerate(images):
                image_number = int(image_no + 1)
                image_metadata[page_num][image_number] = {}

                image_for_gemini, image_name = get_image_for_gemini(
                    doc, image, image_no, image_save_dir, file_name, page_num
                )

                print(
                    f"Extracting image from page: {page_num + 1}, saved as: {image_name}"
                )

                response = get_gemini_response(
                    generative_multimodal_model,
                    model_input=[image_description_prompt, image_for_gemini],
                    safety_settings=safety_settings,
                    stream=True,
                    temperature =image_text_extraction_temperature,
                    max_output_tokens =image_text_extraction_max_output_tokens

                )

                image_embedding = get_image_embedding_from_multimodal_embedding_model(
                    image_uri=image_name,
                    multimodal_embedding_model=multimodal_embedding_model,
                    embedding_size=embedding_size
                )

                image_description_text_embedding = (
                    get_text_embedding_from_text_embedding_model(text=response, text_embedding_model=text_embedding_model)
                )

                image_metadata[page_num][image_number] = {
                    "img_num": image_number,
                    "img_path": image_name,
                    "img_desc": response,
                    # "mm_embedding_from_text_desc_and_img": image_embedding_with_description,
                    "mm_embedding_from_img_only": image_embedding,
                    "text_embedding_from_image_description": image_description_text_embedding,
                }

                # Add sleep to reduce issues with Quota error on API
                if add_sleep_after_page:
                    time.sleep(sleep_time_after_page)
                    print(
                        "Sleeping for ",
                        sleep_time_after_page,
                        """ sec before processing the next page to avoid quota issues. You can disable it: "add_sleep_after_page = False"  """,
                    )
        # Add sleep to reduce issues with Quota error on API
        if add_sleep_after_document:
            time.sleep(sleep_time_after_document)
            print(
                "\n \n Sleeping for ",
                sleep_time_after_document,
                """ sec before processing the next document to avoid quota issues. You can disable it: "add_sleep_after_document = False"  """,
            )

        text_metadata_df = get_text_metadata_df(file_name, text_metadata)
        image_metadata_df = get_image_metadata_df(file_name, image_metadata)

        text_metadata_df_final = pd.concat(
            [text_metadata_df_final, text_metadata_df], axis=0
        )
        image_metadata_df_final = pd.concat(
            [
                image_metadata_df_final,
                image_metadata_df.drop_duplicates(subset=["img_desc"]),
            ],
            axis=0,
        )

        text_metadata_df_final = text_metadata_df_final.reset_index(drop=True)
        image_metadata_df_final = image_metadata_df_final.reset_index(drop=True)

    return text_metadata_df_final, image_metadata_df_final, progress_bar


# Helper Functions


def get_user_query_text_embeddings(user_query: str,text_embedding_model) -> np.ndarray:
    """
    Extracts text embeddings for the user query using a text embedding model.

    Args:
        user_query: The user query text.
        embedding_size: The desired embedding size.

    Returns:
        A NumPy array representing the user query text embedding.
    """

    return get_text_embedding_from_text_embedding_model(user_query, text_embedding_model)


def get_user_query_image_embeddings(
    image_query_path: str, embedding_size: int,multimodal_embedding_model
) -> np.ndarray:
    """
    Extracts image embeddings for the user query image using a multimodal embedding model.

    Args:
        image_query_path: The path to the user query image.
        embedding_size: The desired embedding size.

    Returns:
        A NumPy array representing the user query image embedding.
    """
    print(f"683 {image_query_path}")
    return get_image_embedding_from_multimodal_embedding_model(
        image_uri=image_query_path, embedding_size=embedding_size,multimodal_embedding_model=multimodal_embedding_model
    )


def get_cosine_score(
    dataframe: pd.DataFrame, column_name: str, input_text_embd: np.ndarray
) -> float:
    """
    Calculates the cosine similarity between the user query embedding and the dataframe embedding for a specific column.

    Args:
        dataframe: The pandas DataFrame containing the data to compare against.
        column_name: The name of the column containing the embeddings to compare with.
        input_text_embd: The NumPy array representing the user query embedding.

    Returns:
        The cosine similarity score (rounded to two decimal places) between the user query embedding and the dataframe embedding.
    """

    text_cosine_score = round(np.dot(dataframe[column_name], input_text_embd), 2)
    return text_cosine_score

def get_similar_image_from_query(
    text_metadata_df: pd.DataFrame,
    image_metadata_df: pd.DataFrame,
    text_embedding_model,
    multimodal_embedding_model,
    query: str = "",
    image_query_path: str = "",
    column_name: str = "",
    image_emb: bool = True,
    top_n: int = 3,
    embedding_size: int = 128,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar images from a metadata DataFrame based on a text query or an image query.

    Args:
        text_metadata_df: A Pandas DataFrame containing text metadata associated with the images.
        image_metadata_df: A Pandas DataFrame containing image metadata (paths, descriptions, etc.).
        query: The text query used for finding similar images (if image_emb is False).
        image_query_path: The path to the image used for finding similar images (if image_emb is True).
        column_name: The column name in the image_metadata_df containing the image embeddings or captions.
        image_emb: Whether to use image embeddings (True) or text captions (False) for comparisons.
        top_n: The number of most similar images to return.
        embedding_size: The dimensionality of the image embeddings (only used if image_emb is True).

    Returns:
        A dictionary containing information about the top N most similar images, including cosine scores, image objects, paths, page numbers, text excerpts, and descriptions.
    """
    # Check if image embedding is used
    if image_emb:
        # Calculate cosine similarity between query image and metadata images
        user_query_image_embedding = get_user_query_image_embeddings(
            image_query_path=image_query_path, embedding_size=embedding_size,multimodal_embedding_model=multimodal_embedding_model
        )
        cosine_scores = image_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_image_embedding),
            axis=1,
        )
    else:
        # Calculate cosine similarity between query text and metadata image captions
        user_query_text_embedding = get_user_query_text_embeddings(query, text_embedding_model)
        cosine_scores = image_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_text_embedding),
            axis=1,
        )

    # Remove same image comparison score when user image is matched exactly with metadata image
    cosine_scores = cosine_scores[cosine_scores < 1.0]

    # Get top N cosine scores and their indices
    top_n_cosine_scores = cosine_scores.nlargest(top_n).index.tolist()
    top_n_cosine_values = cosine_scores.nlargest(top_n).values.tolist()

    # Create a dictionary to store matched images and their information
    final_images: Dict[int, Dict[str, Any]] = {}

    for matched_imageno, indexvalue in enumerate(top_n_cosine_scores):
        # Create a sub-dictionary for each matched image
        final_images[matched_imageno] = {}

        # Store cosine score
        final_images[matched_imageno]["cosine_score"] = top_n_cosine_values[
            matched_imageno
        ]

        # Load image from file
        final_images[matched_imageno]["image_object"] = Image.load_from_file(
            image_metadata_df.iloc[indexvalue]["img_path"]
        )

        # Add file name
        final_images[matched_imageno]["file_name"] = image_metadata_df.iloc[indexvalue][
            "file_name"
        ]

        # Store image path
        final_images[matched_imageno]["img_path"] = image_metadata_df.iloc[indexvalue][
            "img_path"
        ]

        # Store page number
        final_images[matched_imageno]["page_num"] = image_metadata_df.iloc[indexvalue][
            "page_num"
        ]

        final_images[matched_imageno]["page_text"] = np.unique(
            text_metadata_df[
                (
                    text_metadata_df["page_num"].isin(
                        [final_images[matched_imageno]["page_num"]]
                    )
                )
                & (
                    text_metadata_df["file_name"].isin(
                        [final_images[matched_imageno]["file_name"]]
                    )
                )
            ]["text"].values
        )

        # Store image description
        final_images[matched_imageno]["image_description"] = image_metadata_df.iloc[
            indexvalue
        ]["img_desc"]

    return final_images


def get_similar_text_from_query(
    query: str,
    text_metadata_df: pd.DataFrame,
    text_embedding_model,
    column_name: str = "",
    top_n: int = 3,
    chunk_text: bool = True,
    print_citation: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar text passages from a metadata DataFrame based on a text query.

    Args:
        query: The text query used for finding similar passages.
        text_metadata_df: A Pandas DataFrame containing the text metadata to search.
        column_name: The column name in the text_metadata_df containing the text embeddings or text itself.
        top_n: The number of most similar text passages to return.
        embedding_size: The dimensionality of the text embeddings (only used if text embeddings are stored in the column specified by `column_name`).
        chunk_text: Whether to return individual text chunks (True) or the entire page text (False).
        print_citation: Whether to immediately print formatted citations for the matched text passages (True) or just return the dictionary (False).

    Returns:
        A dictionary containing information about the top N most similar text passages, including cosine scores, page numbers, chunk numbers (optional), and chunk text or page text (depending on `chunk_text`).

    Raises:
        KeyError: If the specified `column_name` is not present in the `text_metadata_df`.
    """

    if column_name not in text_metadata_df.columns:
        raise KeyError(f"Column '{column_name}' not found in the 'text_metadata_df'")

    query_vector = get_user_query_text_embeddings(query, text_embedding_model)

    # Calculate cosine similarity between query text and metadata text
    cosine_scores = text_metadata_df.apply(
        lambda row: get_cosine_score(
            row,
            column_name,
            query_vector,
        ),
        axis=1,
    )

    # Get top N cosine scores and their indices
    top_n_indices = cosine_scores.nlargest(top_n).index.tolist()
    top_n_scores = cosine_scores.nlargest(top_n).values.tolist()

    # Create a dictionary to store matched text and their information
    final_text: Dict[int, Dict[str, Any]] = {}

    for matched_textno, index in enumerate(top_n_indices):
        # Create a sub-dictionary for each matched text
        final_text[matched_textno] = {}

        # Store page number
        final_text[matched_textno]["file_name"] = text_metadata_df.iloc[index][
            "file_name"
        ]

        # Store page number
        final_text[matched_textno]["page_num"] = text_metadata_df.iloc[index][
            "page_num"
        ]

        # Store cosine score
        final_text[matched_textno]["cosine_score"] = top_n_scores[matched_textno]

        if chunk_text:
            # Store chunk number
            final_text[matched_textno]["chunk_number"] = text_metadata_df.iloc[index][
                "chunk_number"
            ]

            # Store chunk text
            final_text[matched_textno]["chunk_text"] = text_metadata_df["chunk_text"][
                index
            ]
        else:
            # Store page text
            final_text[matched_textno]["text"] = text_metadata_df["text"][index]

    return final_text

def generate_gemini_history_object(chat_list, user_id, bot_id, answer_split_key='Explanation'):
    """
    Generates a history object for Gemini interface from a list of chat interactions.

    This function processes a list of chat dictionaries to create two outputs:
    1. A list representing the history of chat interactions, distinguishing between user and model (bot) roles.
    2. A concatenated string of all queries made during the chat session.

    Parameters:
    - chat_list (list of dict): A list where each element is a dictionary representing a chat message with keys as user_id or bot_id.
    - user_id (str): The identifier for the user in the chat.
    - bot_id (str): The identifier for the bot in the chat.
    - answer_split_key (str): A key to split the bot's response to separate the main answer from further explanation, default is 'Explanation'.

    Returns:
    - tuple: A tuple containing two elements:
        1. A list of Content objects with roles and parts based on the chat messages.
        2. A string of concatenated queries from the chat history.
        Returns (None, '') if the chat history contains fewer than two queries.
    """

    gemini_history = []
    for chat in chat_list:
        if user_id in chat:
            gemini_history.append(Content(role="user", parts=[Part.from_text(chat[user_id])]))
        else:
            gemini_history.append(Content(role="user", parts=[Part.from_text('No question')]))
        if bot_id in chat:
            gemini_history.append(Content(role="model", parts=[Part.from_text(chat[bot_id])]))
        else:
            gemini_history.append(Content(role="model", parts=[Part.from_text('No response')]))
    if len(gemini_history) < 2:
        return None
    else:
        return gemini_history

def chat_backend(query, instruction, text_metadata_df, image_metadata_df, multimodal_model, text_embedding_model, multimodal_embedding_model, loaded_chat_history=None, text_metadata_df_embedding_col="text_embedding_chunk", image_metadata_df_embedding_col="text_embedding_from_image_description",
llm_temperature=1, llm_max_output_tokens = 8192, generate_content = False):
    """
    Processes a multimodal query combining text and image data to generate a response using a multimodal model.

    Parameters:
    - query (str): The user's query or input.
    - instruction (str): Instructions or context to be included in the model's input.
    - text_metadata_df (DataFrame): DataFrame containing metadata about text data.
    - image_metadata_df (DataFrame): DataFrame containing metadata about image data.
    - multimodal_model (Model): The machine learning model that processes multimodal inputs.
    - loaded_chat_history (tuple, optional): Previously loaded chat history, default is None.
    - text_metadata_df_embedding_col (str): Column name in text metadata DataFrame for text embeddings.
    - image_metadata_df_embedding_col (str): Column name in image metadata DataFrame for image description embeddings.
    - temperature (float): The temperature parameter for generation diversity, default is 1.

    Returns:
    - GeminiResponse: A response object generated by the multimodal model based on the processed input.
    """

    matching_results_chunks_data = {}
    matching_results_image_fromdescription_data= {}
    if len(text_metadata_df) > 0:
        matching_results_chunks_data = get_similar_text_from_query(
            query,
            text_metadata_df,
            column_name=text_metadata_df_embedding_col,
            top_n=20,
            chunk_text=True,
            text_embedding_model=text_embedding_model
        )

    if (len(text_metadata_df)) > 0 & (len(image_metadata_df)>0):
        matching_results_image_fromdescription_data = get_similar_image_from_query(
            query=query,
            text_metadata_df=text_metadata_df,
            image_metadata_df=image_metadata_df,
            column_name=image_metadata_df_embedding_col,
            image_emb=False,
            top_n=10,
            embedding_size=1408,
            text_embedding_model = text_embedding_model,
            multimodal_embedding_model = multimodal_embedding_model
        )

    context_text = ["Text Context: "]
    for key, value in matching_results_chunks_data.items():
        context_text.extend([
            "Text Source: ",
            f'file_name: "{value["file_name"]}" Page: "{value["page_num"]}"',
            "Text",
            value["chunk_text"],
        ])

    gemini_content = [
        instruction,
        "Questions: ",
        query,
        "Image Context: ",
    ]
    for key, value in matching_results_image_fromdescription_data.items():
        gemini_content.extend([
            "Image Path: ",
            value["img_path"],
            "Image Description: ",
            value["image_description"],
            "Image:",
            value["image_object"],
        ])
    gemini_content.extend(context_text)

    if generate_content:
        return get_gemini_response(
            multimodal_model,
            model_input=gemini_content,
            stream=True,
            temperature = llm_temperature,
            max_output_tokens = llm_max_output_tokens
        )
    else:
        return get_gemini_chat_response(
            multimodal_model,
            model_input=gemini_content,
            stream=True,
            llm_temperature = llm_temperature,
            llm_max_output_tokens = llm_max_output_tokens
        )

def pdf_to_images(pdf_path, is_read=False):
    """
    Converts a PDF file to a list of images, one for each page.

    Parameters:
    - pdf_path (str or file-like object): The path to the PDF file or a file-like object containing the PDF.
    - is_read (bool): Flag to indicate if the pdf_path is a file-like object that needs to be read, default is False.

    Returns:
    - list: A list of PIL Image objects representing each page of the PDF as an image.
    """
    if is_read:
        doc = pdf_path.read()
        doc = fitz.open("pdf", doc)
    else:
        doc = fitz.open(pdf_path)

    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def compare_images(img1, img2):
    """
    Compares two images and determines if there is any difference between them.

    Parameters:
    - img1 (Image): The first image to compare.
    - img2 (Image): The second image to compare.

    Returns:
    - bool: True if there is a difference between the images, False otherwise.
    """
    diff = PIL.ImageChops.difference(img1, img2)
    return diff.getbbox() is not None

def find_changed_pages(pdf1_path, pdf2_path, pdf1_is_read=False, pdf2_is_read=False):
    """
    Identifies the pages that have changed between two versions of a PDF document.

    Parameters:
    - pdf1_path (str or file-like object): Path or file-like object to the first PDF.
    - pdf2_path (str or file-like object): Path or file-like object to the second PDF.
    - pdf1_is_read (bool): Specifies if the first PDF is already read into memory.
    - pdf2_is_read (bool): Specifies if the second PDF is already read into memory.

    Returns:
    - list: A list of page numbers that have differences between the two PDF versions.
    """
    images1 = pdf_to_images(pdf1_path, pdf1_is_read)
    images2 = pdf_to_images(pdf2_path, pdf2_is_read)

    changed_pages = []
    for i, (img1, img2) in enumerate(zip(images1, images2)):
        if compare_images(img1, img2):
            changed_pages.append(i + 1)  # Page numbers are 1-based

    return changed_pages

def process_file(uploaded_file, pdf_folder_path, text_metadata_df, image_metadata_df, image_description_prompt, model, image_save_dir, text_embedding_model, multimodal_embedding_model, metadat_file_dir, none_chk=False,image_text_extraction_temperature = 0.2, image_text_extraction_max_output_tokens = 2048):
    """
    Processes an uploaded file, checks for changes in pages, updates metadata dataframes, and gets document metadata.

    Parameters:
    - uploaded_file (file-like object): The file uploaded by the user.
    - pdf_folder_path (str): The path to the folder where PDFs are stored.
    - text_metadata_df (DataFrame): DataFrame containing text metadata.
    - image_metadata_df (DataFrame): DataFrame containing image metadata.
    - image_description_prompt (str): Prompt used for generating image descriptions.
    - model (Model): Model used for extracting metadata.
    - none_chk (bool): Flag to check for a specific condition, default is False.

    Returns:
    - tuple: Returns updated text and image metadata DataFrames, and progress bar object.
    """
    pages_changed = None
    if uploaded_file.type != "application/pdf":
        print(f"The uploaded file is not a PDF but {uploaded_file.type}. Please convert it to a PDF before uploading.")
        return None

    if len(text_metadata_df)>0 :
        if uploaded_file.name in text_metadata_df['file_name'].unique():
            print(f"{uploaded_file.name} already exists in the embeddings database.")
            pages_changed = find_changed_pages(os.path.join(pdf_folder_path, uploaded_file.name), uploaded_file, pdf2_is_read=True)
            text_metadata_df = text_metadata_df[~((text_metadata_df['file_name'] == uploaded_file.name) & (text_metadata_df['page_num'].isin(pages_changed)))].reset_index(drop=True)
            image_metadata_df = image_metadata_df[~((image_metadata_df['file_name'] == uploaded_file.name) & (image_metadata_df['page_num'].isin(pages_changed)))].reset_index(drop=True)

    file_path = os.path.join(pdf_folder_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # For BytesIO objects

    uploaded_text_metadata_df, uploaded_image_metadata_df, progress_bar = get_document_metadata(
        generative_multimodal_model = model,  # we are passing gemini 1.5 pro
        pdf_paths = [file_path],
        image_save_dir=image_save_dir,
        image_description_prompt=image_description_prompt,
        text_embedding_model = text_embedding_model,
        multimodal_embedding_model = multimodal_embedding_model,
        embedding_size=1408,
        add_sleep_after_page=True,  # Uncomment this if you are running into API quota issues
        sleep_time_after_page=20,
        add_sleep_after_document=False,  # Uncomment this if you are running into API quota issues
        sleep_time_after_document=20,  # Increase the value in seconds, if you are still getting quota issues.
        none_chk=none_chk,
        pages_changed=pages_changed,
        image_text_extraction_temperature = image_text_extraction_temperature,
        image_text_extraction_max_output_tokens = image_text_extraction_max_output_tokens
    )
    for col in ['text_embedding_page','text_embedding_chunk','mm_embedding_from_img_only','text_embedding_from_image_description']:
        for df_col in [text_metadata_df,uploaded_text_metadata_df,image_metadata_df,uploaded_image_metadata_df]:
            if col in df_col.columns:
                df_col[col] = df_col[col].apply(tuple)
    text_metadata_df = pd.concat([text_metadata_df, uploaded_text_metadata_df], ignore_index=True).reset_index(drop=True).drop_duplicates()
    image_metadata_df = pd.concat([image_metadata_df, uploaded_image_metadata_df], ignore_index=True).reset_index(drop=True).drop_duplicates()
    for col in ['text_embedding_page','text_embedding_chunk','mm_embedding_from_img_only','text_embedding_from_image_description']:
        for df_col in [text_metadata_df,image_metadata_df]:
            if col in df_col.columns:
                df_col[col] = df_col[col].apply(list)
    # Save to a pickle file
    with open(metadat_file_dir, "wb") as f:
        pickle.dump({"text_metadata": text_metadata_df, "image_metadata": image_metadata_df}, f)

    return text_metadata_df, image_metadata_df, progress_bar

def save_chat_history(chat_history,chat_history_file):
    """
    Saves the current state of chat history to a JSON file.

    Parameters:
    - chat_history (list): A list of dictionaries representing the chat history to be saved.
    """
    with open(chat_history_file, 'w') as file:
        json.dump(chat_history, file)

def load_chat_history(chat_history_file):
    """
    Loads the chat history from a JSON file if it exists.

    Returns:
    - list: A list of chat history entries if the file exists and is a list; otherwise, an empty list.
    """
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r') as file:
            chat_history = json.load(file)
            return chat_history if isinstance(chat_history, list) else []
    return []

def update_chat_history(user_input, bot_response, chat_history_file):
    """
    Updates the session state with new chat entries and saves the updated chat history.

    Parameters:
    - user_input (str): The user's input to the chat.
    - bot_response (str): The bot's response to the user's input.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if 'current_chat_history' not in st.session_state:
        st.session_state.current_chat_history = [{"assistant": 'Hi! How can I help you?'}]

    chat_id = len(st.session_state.chat_history)
    st.session_state.chat_history.append({"User": user_input, "assistant": bot_response, "id": chat_id})
    st.session_state.current_chat_history.append({"User": user_input, "assistant": bot_response})
    save_chat_history(st.session_state.chat_history, chat_history_file)

def clear_chat_history(chat_history_file):
    """
    Clears the chat history from the session state and the storage file.
    """
    st.session_state.chat_history = []
    st.session_state.current_chat_history = []
    if 'selected_chat' in st.session_state:
        del st.session_state.selected_chat
    with open(chat_history_file, 'w') as file:
        json.dump([], file)
    st.experimental_rerun()

def get_base64(bin_file):
    """
    Encodes a binary file into a base64 string.

    Parameters:
    - bin_file (str): Path to the binary file.

    Returns:
    - str: The base64 encoded string of the file's contents.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    """
    Sets the background of a Streamlit app using a PNG file.

    Parameters:
    - png_file (str): Path to the PNG file.
    """
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
      background-image: url("data:image/png;base64,{bin_str}");
      background-size: cover;
    }}
    [data-testid="stBottom"] > div {{
        background: transparent;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def replace_color(img_path, old_color, new_color):
    """
    Replaces a specified color in an image with a new color.

    Parameters:
    - img_path (str): Path to the image file.
    - old_color (tuple): The RGB tuple of the color to replace.
    - new_color (tuple): The RGB tuple of the new color.

    Returns:
    - Image: The modified image with the color replaced.
    """
    with PIL.Image.open(img_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pixels = img.load()
        for y in range(img.height):
            for x in range(img.width):
                if pixels[x, y] == old_color:
                    pixels[x, y] = new_color
        return img

def authenticate_vertexai(cred_file_path, project_id, location):
    """
    Initializes and authenticates Vertex AI with the provided credentials, project ID, and location.

    Parameters:
    - credentials: Credentials object used for authentication.
    - project_id (str): Google Cloud project ID.
    - location (str): Google Cloud location to initialize Vertex AI services.
    """
    credentials = service_account.Credentials.from_service_account_file(cred_file_path)
    vertexai.init(project=project_id, location=location, credentials=credentials)
    return credentials


def transcribe_audio(audio, credentials) -> speech.RecognizeResponse:
    """Transcribe the given audio."""
    wav = wave.open(io.BytesIO(audio))
    client = speech.SpeechClient(credentials = credentials)
    audio = speech.RecognitionAudio(content=audio)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=wav.getframerate(),
        audio_channel_count = wav.getnchannels(),
        language_code="en-US"
    )

    response = client.recognize(config=config, audio=audio)
    transcription = []
    for result in response.results:
        transcription.append(result.alternatives[0].transcript)
    return " ".join(transcription)
