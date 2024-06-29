# GPT-4o (2024) versus ELIZA (1966)

## Introduction

One day when I was bored, I wondered what a conversation between [GPT-4o](https://en.wikipedia.org/wiki/ChatGPT) (2024), the world-famous AI, and [ELIZA](https://en.wikipedia.org/wiki/ELIZA) (1964), one of the first (the first?) conversational AI in the history of computing, would look like.
If you're a somewhat older geek, you probably know ELIZA! It's the "doctor" integrated into Emacs, but also the basis used by spam bots on ICQ (You know, ladies who generously offered us to discover them a little more on websites).

ELIZA, only available in English, only plays the role of a psychiatrist. It operates by rephrasing the patient's statements to give the illusion of understanding.

GPT, which you undoubtedly know, is much more versatile and is capable of discussing anything.

## Which file to read

There are several files available in both French and English.

### French

- [La version uniquement avec le texte](only_text_fr.md)
- [La version avec le code](eliza-gpt-fr.md)

### English

- [Text only version](only_text_en.md)
- [The Python notebook (with code)](eliza-gpt.ipynb)

## How to Try It Yourself

To try this notebook yourself, follow these steps:

1. **Download the necessary files**: Download the zip file from the provided gist link and extract it. The zip file should contain `eliza-gpt.ipynb`, `eliza.py`, and `doctor.txt`.

2. **Set up your environment**: Create a virtual environment and install the required dependencies by running:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Create a .env file**: In the root directory of your project, create a `.env` file and define the `OPENAI_API_KEY` variable. The file should look like this:

    ```.env
    OPENAI_API_KEY=your_openai_api_key
    ```

4. **Run the notebook**: Start Jupyter Notebook or JupyterLab and open the notebook file (`eliza-gpt.ipynb`). You can do this by running:

    ```bash
    jupyter notebook
    ```

    or

    ```bash
    jupyter lab
    ```

5. **Execute the cells**: In the notebook, execute the cells one by one to see the results. Make sure you follow any specific instructions provided in the notebook to properly configure and run the code.

6. **Explore and modify**: Feel free to explore the code and modify it to suit your needs. You can experiment with different configurations and inputs to see how the ELIZA and GPT-4 models respond in various scenarios.

If you encounter any issues or have questions, refer to the documentation provided within the notebook.

## Credits

- ELIZA by Joseph Weizenbaum
- eliza.py: A Python implementation of [ELIZA by wadetb](https://github.com/wadetb/eliza)
- GPT-4o from [OpenAI](https://openai.com)
