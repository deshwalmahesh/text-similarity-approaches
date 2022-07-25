import uvicorn
from fastapi import FastAPI, HTTPException, Form # pip install fastapi
import traceback
# import contractions # pip install contractions
import re

from sentence_transformers import SentenceTransformer, util # pip install sentence_transformers
model = SentenceTransformer('paraphrase-albert-small-v2') # Smallest Model of 46 MB, can use any

app = FastAPI(title='Sentence Similarity API')


def preprocess(string:str)->str:
    '''
    There are some Assumptions for the inputs. 
    Given which kind of data we're handling we can build our own processing function
    '''
    string = string.lower()
    # string = contractions.fix(string) # remove contractions
    string = re.sub('[^a-z]', ' ', string)
    return re.sub('\s+', ' ', string)


@app.post('/predict')
async def similarity(sentence1:str = Form(...), sentence2:str = Form(...)):
    """
    Get two sentences and send a score between [0,1]. 0 means no similarity at all, 1 means exactly same sentence
    """
    try:
        # Preprocessing depends on the "TYPE" of data we're supposed to handle so skipping for now

        embeddings1 = model.encode(sentence1) #Compute embedding for both lists
        embeddings2 = model.encode(sentence2)
        return round(util.cos_sim(embeddings1, embeddings2).item(), 3) # Return Cosine Similarity between [0,1]

    except Exception as e:
        e = traceback.format_exc() # just for backend, you can save it in log file
        raise HTTPException(status_code=420, detail=f"Internal Error :: {e}") # custom error

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True, debug = True, host = '0.0.0.0', port = 5678)
