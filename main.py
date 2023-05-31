import argparse
import asyncio

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types import KeyboardButton
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from pathlib import Path
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings

THIS_DIR = Path(__file__).parent

MAX_MESSAGE_LENGTH = 4000
import os


# bot_token = os.environ.get('TELEGRAM_API_KEY')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--telegram_token", help="Telegram bot token", type=str, required=True
    )
    return parser.parse_args()


# Create datastore
if os.path.exists("data_store"):
    vector_store = FAISS.load_local(
        "data_store",
        OpenAIEmbeddings()
    )
else:
    file = "kb.pdf"
    loader = PyPDFLoader(file)
    input_text = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    print(input_text)
    vector_store = FAISS.from_documents(input_text, embeddings)
    # Save the files `to local disk.
    vector_store.save_local("data_store")

print("Index is built")

args = parse_args()
# Set up the Telegram bot
bot = Bot(token=args.telegram_token)
dispatcher = Dispatcher(bot)

# Define a ReplyKeyboardMarkup to show a "start" button
RESTART_KEYBOARD = types.ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton('/start')]], resize_keyboard=True, one_time_keyboard=True
)

system_template = """You are Trade Helper Assistant. Use the following pieces of context to answer the users 
question. Don't mention the source file path in any way. In order to help traders to better understand how to 
integrate with our futures trading system, we have made the “VIP Trader Handbook - Futures Trading”.  In this 
handbook we have listed all the frequent asked questions and their answers, from contract specifications and trading 
rules, to exchange infrastructure and API usage. Do not answer questions that are not related to trading topics. Keep 
it on the topic of trading very strictly. Your knowledge is limited to topics related to trading, blockchain, 
smart contracts, and AI. If you don't know the answer, just say that "I don't know", don't try to make up an answer. 
Answer only about trading related questions. If question is not related to trading, just say that "I'm sorry, 
but that topic is not related to trading, blockchain, smart contracts, or AI. My knowledge is limited to those 
topics, so I won't be able to provide a relevant answer. Is there anything else related to trading or the 
aforementioned topics that I can help you with?". ---------------- {summaries} """
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

chain_type_kwargs = {"prompt": prompt}
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1,
                 max_tokens=256)  # Modify model_name if you have access to GPT-4


# Define a handler for the "/start" command
@dispatcher.message_handler(commands=["start"])
async def start(message: types.Message):
    # Show a "typing" action to the user
    await bot.send_chat_action(message.from_user.id, action=types.ChatActions.TYPING)

    # Send a welcome message with a "start" button
    await bot.send_message(
        message.from_user.id,
        text="Hi there!\nI'm Trade Helper. How can I assist you today?",
        # reply_markup=RESTART_KEYBOARD
    )
    # Pause for 1 second
    await asyncio.sleep(1)


# Define the handler function for the /query command
@dispatcher.message_handler()
async def handle_query_command(message: types.Message):
    await bot.send_chat_action(
        message.from_user.id, action=types.ChatActions.TYPING
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    result_text = chain(message.text)

    num_messages = len(result_text['answer']) // MAX_MESSAGE_LENGTH
    await bot.send_chat_action(
        message.from_user.id, action=types.ChatActions.TYPING
    )
    for i in range(num_messages + 1):
        await bot.send_chat_action(
            message.from_user.id, action=types.ChatActions.TYPING
        )
        await asyncio.sleep(1)
        await bot.send_message(
            message.from_user.id,
            text=result_text['answer'][i * MAX_MESSAGE_LENGTH: (i + 1) * MAX_MESSAGE_LENGTH],
        )


# Start polling for updates from Telegram
if __name__ == "__main__":
    executor.start_polling(dispatcher, skip_updates=False)
