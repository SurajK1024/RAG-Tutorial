# RAG - Retrieval-Augmented Generation

### Retrieval
- Fetch the proper context from the document/knowledge base.

### Generation 
- Create the text from the given/specific retrieved contexts.

#### Problem Statement
1. How to keep knowledge base?
2. How to Retrieve proper context?
3. Where to store the knowledge base?
4. Difficulties of storing the knowledge base?
5. Difficulties of Generating the Results?
6. What is hallucination?
7. Knowledge base as Text, Image and Tables?

### Prompt Engineering
#### PICFARTD
  - PERSONA > give identity to the LLM
  - INSTRUCTION > provide the task 
  - CONTEXT > additional information
  - FORMAT > how is the way the LLM should work on the text generation
  - AUDIENCE > whom to target
  - TONE > Way of delivery the text
  - DATA > summary

## Lets Start
1. Data gathering and processing
- We have to go through each type of data to store in the knowledge base.
- If the data is simple text then we can easily use Library `PyMuPDF` to extract it.
- Same Library can be use to extract image as a one full image.
- but to find the text inside the image, we have to think out of the box.
- OCR -> Optical Character Recognization
- Now for Tabluar data (docling) help to convert table into text preserving the rows and columns
- WebScrapping -> PUPPETEER
