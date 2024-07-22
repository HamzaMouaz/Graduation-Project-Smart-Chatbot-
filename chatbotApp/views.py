from django.shortcuts import render, redirect
from django.contrib import auth, messages
from .forms import LoginForm, SignUpForm
from django.contrib.auth import authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .models import Conversation as Conv
from .models import Userr
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from django.http import JsonResponse
import torch
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import nltk
nltk.download('punkt')
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from django.http import HttpResponse, request  # Assurez-vous d'importer le module request depuis django.http
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from huggingface_hub import login
import os

from huggingface_hub import notebook_login


token = "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"
device_map = {"": 0}
from huggingface_hub import notebook_login
notebook_login()
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    token=token
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,token=token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_llm = SentenceTransformerEmbeddings(model_name=model_name)


@login_required
def chatbot_view(request):
    user_id = request.user.id
    user = Userr.objects.get(id=user_id)
    user_name = Userr.objects.get(id=user_id).username
    user_conversations = Conv.objects.filter(user=user)
    return render(request, 'chat.html', {'user_name': user_name, 'user_conversations':user_conversations})



def getResponse(request):
    userMessage = request.GET.get('userMessage')
    user_id = request.user.id
    user = Userr.objects.get(id=user_id)
    user_name = Userr.objects.get(id=user_id).username
    user_conversations = Conv.objects.filter(user=user)
    # Récupérer l'historique du chat de l'utilisateur
    history = [conv.message for conv in user_conversations]
    # Récupérer l'historique du chat de l'utilisateur
    history = [conv.message for conv in user_conversations]
    #################
    llm_loader = PyPDFLoader("chatbotApp/static/file.pdf")
    pages = llm_loader.load_and_split()


    text_splitter = CharacterTextSplitter(
    separator="separatoor",
    is_separator_regex=False,
    # Utilisez le séparateur de votre choix
    chunk_size=200,
    chunk_overlap=0
    )

    documents=["".join([pages[i].page_content for i in range(len(pages))])]
    metadatas = [ {"document":"a propos de djezzy"}]

    tokens_chunks = text_splitter.create_documents(
    documents, metadatas=metadatas
    )

    
    docs_text = [ chunk.page_content for chunk in tokens_chunks ]
    docs_embeddings = embedding_llm.embed_documents(docs_text)

    query_text = "what are the benefits for Hadj and Omra service 2,000 DA Package?"
    query_embedding = embedding_llm.embed_query(query_text)
    query_embedding_array = np.array(query_embedding)
    docs_embeddings=np.array(docs_embeddings)
    similarities = [cosine_similarity(doc.reshape(1,-1), query_embedding_array.reshape(1,-1)) for doc in docs_embeddings]
    sorted_docs = sorted(zip(docs_text, docs_embeddings, similarities), key=lambda x: x[2], reverse=True)
    similar_docs = [(doc,sim) for doc, _, sim in sorted_docs ]


        # Filtrer les documents avec une similarité supérieure à 0.4
    similar_docs1 = [(doc,sim) for doc, _, sim in sorted_docs if sim > 0.72]
    if  not similar_docs1:
      similar_docs2 = [(doc,sim) for doc, _, sim in sorted_docs if sim > 0.65]
      if  not similar_docs2:
        similar_docs = [(doc,sim) for doc, _, sim in sorted_docs if sim > 0.4]
        if  not similar_docs:
          similar_docsA = [(doc,sim) for doc, _, sim in sorted_docs if (sim >= 0.3 and sim<0.4)]
          if  not similar_docsA:
            result="As a chatbot for Djezzy, I can provide information exclusively about our affiliated companies. Unfortunately, I'm unable to respond to inquiries outside of that scope."
            chatResponse= result
          else:
            result="I apologize, I don't fully understand your question. You can contact our customer service for answers to your needs, or if you can provide more details, I would be happy to help."
            chatResponse= result
        else:
          context="\n---------------------\n".join([doc for doc,_ in similar_docs[:3]]if len(similar_docs) >=3 else [doc for doc, _ in similar_docs[:1]])

          system_message=" "
          prompt = f" <bos><start_of_turn>user\n When answering questions, please follow these instructions:\nRead each paragraph in the context provided.\nDetermine which paragraphs, if any, are relevant to answering the question asked. Ignore paragraphs that are not related to the question.\nIf there is no paragraph that is relevant to answering the question, simply state: For this question, it's best to reach out to our customer service team. They'll be able to assist you with your needs.\nIf there are relevant paragraphs, use the information in them to provide a concise answer to the question. Do not include any additional details beyond the direct answer.\nProvide only the answer, without any other commentary or explanation.\n###context:\n{context}\n###question:\n{query_text}\n###answer:\n<end_of_turn>\n <start_of_turn>model" # replace the command here with something relevant to your task
          pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2500, do_sample= True, temperature= 0.1)
          result = pipe(prompt)
          result=result[0]['generated_text']
          parts = result.split("###answer:")
          chatResponse= parts[1]
      else:
        context = "\n---------------------\n".join([doc for doc, _ in similar_docs2[:2]] if len(similar_docs2) >= 2 else [doc for doc, _ in similar_docs2[:1]])

        system_message=" "
        prompt = f" <bos><start_of_turn>user\n When answering questions, please follow these instructions:\nRead each paragraph in the context provided.\nDetermine which paragraphs, if any, are relevant to answering the question asked. Ignore paragraphs that are not related to the question.\nIf there is no paragraph that is relevant to answering the question, simply state: For this question, it's best to reach out to our customer service team. They'll be able to assist you with your needs.\nIf there are relevant paragraphs, use the information in them to provide a concise answer to the question. Do not include any additional details beyond the direct answer.\nProvide only the answer, without any other commentary or explanation.\n###context:\n{context}\n###question:\n{query_text}\n###answer:\n<end_of_turn>\n <start_of_turn>model"
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2500, do_sample= True, temperature= 0.1)
        result = pipe(prompt)
        result=result[0]['generated_text']
        parts = result.split("###answer:")
        chatResponse= parts[1]

    else:
      context="\n---------------------\n".join([doc for doc,_ in similar_docs1[:1]])
      system_message=" "
      prompt = f" <bos><start_of_turn>user\n When answering questions, please follow these instructions:\nRead each paragraph in the context provided.\nDetermine which paragraphs, if any, are relevant to answering the question asked. Ignore paragraphs that are not related to the question.\nIf there is no paragraph that is relevant to answering the question, simply state: For this question, it's best to reach out to our customer service team. They'll be able to assist you with your needs.\nIf there are relevant paragraphs, use the information in them to provide a concise answer to the question. Do not include any additional details beyond the direct answer.\nProvide only the answer, without any other commentary or explanation.\n###context:\n{context}\n###question:\n{query_text}\n###answer:\n<end_of_turn>\n <start_of_turn>model" # replace the command here with something relevant to your task
      pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000, do_sample= True, temperature= 0.1)
      result = pipe(prompt)
      result=result[0]['generated_text']
      parts = result.split("###answer:")
      chatResponse= parts[1]
    Conv.objects.create(user=user, message=userMessage, response= chatResponse)
    return HttpResponse(chatResponse,user_name)



def index(request):
    return render(request, 'index.html')

def non(request):
    return render(request, 'non.html')

def view_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = auth.authenticate(username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            messages.error(request, 'Invalid login details')
            #return redirect('non')
    return render(request, 'login1.html', {'form': LoginForm})

def logout(request):
    auth.logout(request)
    messages.info(request, 'You have been logged out!!')
    return redirect('/')

def historique(request):
    user_id = request.user.id
    user = Userr.objects.get(id=user_id)
    user_name = Userr.objects.get(id=user_id).username
    user_conversations = Conv.objects.filter(user=user)
    return render(request, 'historiq.html', {'user_name': user_name, 'user_conversations':user_conversations})


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            #view_login(request,user)
            return redirect('index')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})
    