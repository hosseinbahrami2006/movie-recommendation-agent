# Movie Recommendation Agent

A command-line movie recommendation application built with **LangGraph**, **LangChain**, **Ollama**, and **DuckDuckGo Search**.

The application collects your movie preferences, searches the web for relevant information, and uses a locally running language model to recommend five movies. It then searches for IMDb and Netflix links and displays a formatted report.

## Features

- Generates five movie recommendations based on detailed preferences.
- Supports optional age, gender, and country information.
- Searches the web before generating recommendations.
- Uses a local Ollama model: `qwen2.5:3b`.
- Requests structured JSON output from the model.
- Validates recommendations and removes duplicate titles.
- Retries generation up to three times when output validation fails.
- Searches for IMDb and Netflix links for each movie.
- Provides fallback search links when no links are found.
- Uses randomized generation settings to encourage varied recommendations.
- Produces movie titles, descriptions, and recommendation reasons in English.

> Netflix availability is not verified. Search results and links may be incomplete, outdated, or unrelated to the exact movie.

## How It Works

The application uses a fixed, four-step LangGraph workflow:

```text
START
  |
  v
Collect User Profile
  |
  v
Search the Web and Generate Movie Candidates
  |
  v
Search for Movie Links
  |
  v
Build Final Answer
  |
  v
END
```

### 1. Collect user information

The application asks for:

- Age — optional.
- Gender — optional.
- Country — optional, used in availability-related searches.
- Detailed movie preferences — recommended for better results.

Preferences can describe atmosphere, pacing, characters, themes, visual style, emotions, or endings.

### 2. Generate recommendations

The application searches DuckDuckGo using the user's preferences and passes the results to the local model as reference information.

The model is instructed to return five real movies in JSON format, including:

- Title.
- Release year.
- Short description.
- Reason for the recommendation.

The application checks the JSON structure and requires at least five valid, distinct titles. If validation fails, it retries up to three times.

### 3. Search for links

For each recommended movie, the application searches for IMDb and Netflix links.

If no matching-domain links are extracted, the final report includes search URLs instead.

### 4. Display the report

The application prints a Markdown-formatted report to the terminal.

## Requirements

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and running.
- The `qwen2.5:3b` model downloaded.
- An internet connection for DuckDuckGo searches.

No paid API key is required.

## Project Structure

```text
movie-recommendation-agent/
├── main.py             # Application code and LangGraph workflow
├── README.md           # Documentation
└── requirements.txt    # Python dependencies
```

## Installation

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-agent.git
cd movie-recommendation-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS and Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from:

https://ollama.com/

If the Ollama service is not already running, start it in a separate terminal:

```bash
ollama serve
```

### 5. Download the model

```bash
ollama pull qwen2.5:3b
```

Confirm that the model is available:

```bash
ollama list
```

## Usage

Run the application:

```bash
python main.py
```

Example input:

```text
What is your age? (optional): 25
What is your gender? (optional):
Which country do you live in? (used for availability-related searches): Canada

Describe the kinds of movies you enjoy.
You can mention atmosphere, story structure, characters, emotions,
pacing, themes, visual style, and endings:
> I enjoy slow-burning psychological mysteries with complex characters,
  atmospheric cinematography, and unexpected but believable endings.
```

Press Enter to skip a question. If no preferences are supplied, the model is instructed to provide varied recommendations.

### Output Format

The final report includes:

```text
# Movie Recommendations

## User Profile
- Age: ...
- Gender: ...
- Country: ...
- Preferences: ...

## Recommended Movies

### 1. Movie Title (Year)

Description: ...
Why it matches: ...

IMDb: ...
Netflix: ...
```

The report is printed as Markdown text; the application does not render it as a web page or save it to a file.

## Configuration

Model settings are defined near the beginning of `main.py`:

```python
MODEL_NAME = "qwen2.5:3b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    repeat_penalty=1.1,
    seed=RANDOM_SEED,
    num_predict=1800,
    format="json",
)
```

### Use a different model

Download the desired model:

```bash
ollama pull qwen2.5:7b
```

Then update:

```python
MODEL_NAME = "qwen2.5:7b"
```

Larger models may improve recommendation quality and JSON reliability, but require more memory and processing time.

### Adjust variation

- Lower `temperature` for more consistent output.
- Higher `temperature` can increase variety but may reduce reliability.
- The application also randomizes its seed and recommendation variation instructions.

These settings encourage variety; they do not guarantee different recommendations on every run.

## Dependencies

| Package | Purpose |
| --- | --- |
| `langchain-community` | DuckDuckGo search tool integration |
| `langchain-ollama` | Interface to locally running Ollama models |
| `langgraph` | Stateful workflow orchestration |
| `ddgs` | Search backend used by newer DuckDuckGo integrations |
| `duckduckgo-search` | Compatibility with integrations using the older backend |

Modules such as `json`, `random`, `re`, `typing`, and `urllib.parse` are included in Python's standard library.

## Limitations

- Recommendations and movie details are generated by a language model and may contain factual errors.
- Validation checks structure and duplicate titles; it does not independently verify movie existence, release years, or plot details.
- Search results do not guarantee that recommendations accurately match the user's preferences.
- Some DuckDuckGo tool versions return mainly text rather than source URLs, so fallback links may be used frequently.
- Extracted links are checked for IMDb or Netflix domains, but are not verified as exact movie pages.
- Netflix availability depends on country and can change over time.
- The program normally performs six searches per successful run: one preference search and five movie-link searches.
- Search services may impose rate limits or temporarily fail.
- Age is included in the prompt, but the application does not enforce content ratings or age suitability.
- The workflow is predefined; the model does not autonomously choose tools or change the execution path.
- A local model is used for generation, but search queries are sent to an external search service.

## Troubleshooting

### Cannot connect to Ollama

Ensure Ollama is running:

```bash
ollama serve
```

### Model not found

Download the configured model:

```bash
ollama pull qwen2.5:3b
```

### DuckDuckGo search fails

Check your internet connection and try again later if requests are rate-limited.

The application continues using the model's knowledge if the initial preference search fails. If a movie-link search fails, it provides fallback search URLs.

### Invalid JSON or fewer than five movies

The application retries generation automatically. If all three attempts fail:

- Try running the application again.
- Lower the model temperature.
- Use a larger model.
- Provide clearer movie preferences.

### Generation is slow

Generation speed depends on your hardware, model size, prompt length, and Ollama configuration. Web searches also add latency.
# Movie Recommendation Agent

A command-line movie recommendation application built with **LangGraph**, **LangChain**, **Ollama**, and **DuckDuckGo Search**.

The application collects your movie preferences, searches the web for relevant information, and uses a locally running language model to recommend five movies. It then searches for IMDb and Netflix links and displays a formatted report.

## Features

- Generates five movie recommendations based on detailed preferences.
- Supports optional age, gender, and country information.
- Searches the web before generating recommendations.
- Uses a local Ollama model: `qwen2.5:3b`.
- Requests structured JSON output from the model.
- Validates recommendations and removes duplicate titles.
- Retries generation up to three times when output validation fails.
- Searches for IMDb and Netflix links for each movie.
- Provides fallback search links when no links are found.
- Uses randomized generation settings to encourage varied recommendations.
- Produces movie titles, descriptions, and recommendation reasons in English.

> Netflix availability is not verified. Search results and links may be incomplete, outdated, or unrelated to the exact movie.

## How It Works

The application uses a fixed, four-step LangGraph workflow:

```text
START
  |
  v
Collect User Profile
  |
  v
Search the Web and Generate Movie Candidates
  |
  v
Search for Movie Links
  |
  v
Build Final Answer
  |
  v
END
```

### 1. Collect user information

The application asks for:

- Age — optional.
- Gender — optional.
- Country — optional, used in availability-related searches.
- Detailed movie preferences — recommended for better results.

Preferences can describe atmosphere, pacing, characters, themes, visual style, emotions, or endings.

### 2. Generate recommendations

The application searches DuckDuckGo using the user's preferences and passes the results to the local model as reference information.

The model is instructed to return five real movies in JSON format, including:

- Title.
- Release year.
- Short description.
- Reason for the recommendation.

The application checks the JSON structure and requires at least five valid, distinct titles. If validation fails, it retries up to three times.

### 3. Search for links

For each recommended movie, the application searches for IMDb and Netflix links.

If no matching-domain links are extracted, the final report includes search URLs instead.

### 4. Display the report

The application prints a Markdown-formatted report to the terminal.

## Requirements

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and running.
- The `qwen2.5:3b` model downloaded.
- An internet connection for DuckDuckGo searches.

No paid API key is required.

## Project Structure

```text
movie-recommendation-agent/
├── main.py             # Application code and LangGraph workflow
├── README.md           # Documentation
└── requirements.txt    # Python dependencies
```

## Installation

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-agent.git
cd movie-recommendation-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS and Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from:

https://ollama.com/

If the Ollama service is not already running, start it in a separate terminal:

```bash
ollama serve
```

### 5. Download the model

```bash
ollama pull qwen2.5:3b
```

Confirm that the model is available:

```bash
ollama list
```

## Usage

Run the application:

```bash
python main.py
```

Example input:

```text
What is your age? (optional): 25
What is your gender? (optional):
Which country do you live in? (used for availability-related searches): Canada

Describe the kinds of movies you enjoy.
You can mention atmosphere, story structure, characters, emotions,
pacing, themes, visual style, and endings:
> I enjoy slow-burning psychological mysteries with complex characters,
  atmospheric cinematography, and unexpected but believable endings.
```

Press Enter to skip a question. If no preferences are supplied, the model is instructed to provide varied recommendations.

### Output Format

The final report includes:

```text
# Movie Recommendations

## User Profile
- Age: ...
- Gender: ...
- Country: ...
- Preferences: ...

## Recommended Movies

### 1. Movie Title (Year)

Description: ...
Why it matches: ...

IMDb: ...
Netflix: ...
```

The report is printed as Markdown text; the application does not render it as a web page or save it to a file.

## Configuration

Model settings are defined near the beginning of `main.py`:

```python
MODEL_NAME = "qwen2.5:3b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    repeat_penalty=1.1,
    seed=RANDOM_SEED,
    num_predict=1800,
    format="json",
)
```

### Use a different model

Download the desired model:

```bash
ollama pull qwen2.5:7b
```

Then update:

```python
MODEL_NAME = "qwen2.5:7b"
```

Larger models may improve recommendation quality and JSON reliability, but require more memory and processing time.

### Adjust variation

- Lower `temperature` for more consistent output.
- Higher `temperature` can increase variety but may reduce reliability.
- The application also randomizes its seed and recommendation variation instructions.

These settings encourage variety; they do not guarantee different recommendations on every run.

## Dependencies

| Package | Purpose |
| --- | --- |
| `langchain-community` | DuckDuckGo search tool integration |
| `langchain-ollama` | Interface to locally running Ollama models |
| `langgraph` | Stateful workflow orchestration |
| `ddgs` | Search backend used by newer DuckDuckGo integrations |
| `duckduckgo-search` | Compatibility with integrations using the older backend |

Modules such as `json`, `random`, `re`, `typing`, and `urllib.parse` are included in Python's standard library.

## Limitations

- Recommendations and movie details are generated by a language model and may contain factual errors.
- Validation checks structure and duplicate titles; it does not independently verify movie existence, release years, or plot details.
- Search results do not guarantee that recommendations accurately match the user's preferences.
- Some DuckDuckGo tool versions return mainly text rather than source URLs, so fallback links may be used frequently.
- Extracted links are checked for IMDb or Netflix domains, but are not verified as exact movie pages.
- Netflix availability depends on country and can change over time.
- The program normally performs six searches per successful run: one preference search and five movie-link searches.
- Search services may impose rate limits or temporarily fail.
- Age is included in the prompt, but the application does not enforce content ratings or age suitability.
- The workflow is predefined; the model does not autonomously choose tools or change the execution path.
- A local model is used for generation, but search queries are sent to an external search service.

## Troubleshooting

### Cannot connect to Ollama

Ensure Ollama is running:

```bash
ollama serve
```

### Model not found

Download the configured model:

```bash
ollama pull qwen2.5:3b
```

### DuckDuckGo search fails

Check your internet connection and try again later if requests are rate-limited.

The application continues using the model's knowledge if the initial preference search fails. If a movie-link search fails, it provides fallback search URLs.

### Invalid JSON or fewer than five movies

The application retries generation automatically. If all three attempts fail:

- Try running the application again.
- Lower the model temperature.
- Use a larger model.
- Provide clearer movie preferences.

### Generation is slow

Generation speed depends on your hardware, model size, prompt length, and Ollama configuration. Web searches also add latency.

## Educational Purpose

This project demonstrates:

- Defining state with `TypedDict`.
- Building a multi-step LangGraph workflow.
- Integrating a local language model.
- Incorporating web search results into prompts.
- Parsing and validating JSON output.
- Handling generation and search failures.
- Producing a formatted recommendation report.
## Educational Purpose

This project demonstrates:

- Defining state with `TypedDict`.
- Building a multi-step LangGraph workflow.
- Integrating a local language model.
- Incorporating web search results into prompts.
- Parsing and validating JSON output.
- Handling generation and search failures.
- Producing a formatted recommendation report.
