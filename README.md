# GPT-5 Testing Dashboard - Python & Streamlit

![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=for-the-badge&logo=streamlit)

A comprehensive Streamlit-based testing dashboard for OpenAI's GPT-5 API, providing systematic evaluation of all model variants, advanced parameter testing, and detailed performance analytics. 

Read the comprehensive blog post: [**GPT-5 API Testing: Building GPT-5 API Testing Dashboard with Streamlit**](https://walterpinem.com/gpt-5-api-testing-streamlit/).

---

## 🚀 Features

### Core Testing Capabilities
- **Multi-Model Support**: Test all GPT-5 variants (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat-latest)
- **Advanced Parameter Testing**: Reasoning effort, verbosity controls, and custom tools
- **Real-Time Progress Tracking**: Live updates during API calls with detailed status information
- **Comprehensive Error Handling**: Robust response parsing with fallback strategies

### Testing Categories
- **Basic Tests**: Connectivity validation and model comparison
- **New Features**: Reasoning effort and verbosity parameter exploration
- **Capabilities**: Coding tests, instruction following, and factual accuracy
- **Performance**: Speed comparison and token efficiency analysis

### Analytics & Visualization
- **Interactive Dashboard**: Real-time performance metrics and trend analysis
- **Advanced Charts**: Plotly-powered visualizations for response time, token usage, and efficiency
- **Export Options**: JSON export for test results
- **Session Management**: Persistent result tracking across testing sessions

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key with GPT-5 access
- 4GB+ available RAM for optimal performance

### Quick Start

1. **Save the code**
   Save the provided `gpt-5-testing.py` file to your local directory

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit openai pandas plotly python-dotenv requests
   ```

4. **Set up environment variables (optional)**
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run gpt-5-testing.py
   ```

The dashboard will open in your browser at `http://localhost:8501`

## 🔧 Configuration

### API Key Setup

You can provide your OpenAI API key in two ways:
1. **Environment variable**: Set `OPENAI_API_KEY` in your environment
2. **Direct input**: Enter your key in the sidebar when running the app

### Optional Environment Variables

```bash
export OPENAI_API_KEY=your_openai_api_key_here
export LOG_LEVEL=INFO
export MAX_CONCURRENT_TESTS=3
export RESULTS_RETENTION_DAYS=7
```

## 📊 Usage Guide

### Available Test Categories

#### Basic Tests
- **Basic Connectivity**: Validates API access across all GPT-5 models
- **Model Comparison**: Side-by-side performance analysis

#### New Features
- **Reasoning Effort**: Tests minimal, low, medium, and high reasoning levels
- **Verbosity Controls**: Compares low, medium, and high verbosity outputs
- **Custom Tools**: (Framework provided for future implementation)
- **Allowed Tools**: (Framework provided for future implementation)

#### Capabilities
- **Coding Tests**: Frontend generation, bug fixing, SQL generation, algorithm implementation
- **Instruction Following**: (Framework provided)
- **Factual Accuracy**: (Framework provided)

#### Performance
- **Speed Comparison**: (Framework provided)
- **Token Efficiency**: (Framework provided)

### Custom Testing

The custom test tab allows you to:
- Select any GPT-5 model variant
- Configure reasoning effort (minimal, low, medium, high)
- Set verbosity levels (low, medium, high)
- Specify custom test names
- Set maximum output tokens
- Enter custom prompts

### Results Dashboard

View comprehensive analytics including:
- Total tests run and success rates
- Average response times and token usage
- Interactive charts showing performance trends
- Detailed results table with filtering options
- JSON export functionality

## 🏗️ Code Structure

### Main Components

```
gpt-5-testing.py                 # Complete Streamlit application (single file)
```

### Key Classes and Functions

- **`GPT5StreamlitTester`**: Core testing framework class
- **`TestResult`**: Dataclass for storing test outcomes
- **`make_responses_api_call()`**: Handles GPT-5 Responses API integration
- **`extract_response_content()`**: Multi-layered response parsing
- **`display_result()`**: Formats and displays test results

## 📈 Performance Metrics Tracked

| Metric | Description |
|--------|-------------|
| Response Time | API call latency in seconds |
| Token Usage | Input/output token consumption |
| Success Rate | Percentage of successful API calls |
| Word Count | Length of generated responses |
| Test Parameters | All API parameters used for each test |

## 🔒 Security Features

- Password-type input for API keys (not stored permanently)
- Session-based API key management
- No persistent storage of sensitive data
- Built-in error handling for API failures

## 🐛 Troubleshooting

### Common Issues

**"Required packages not installed" Error**
```bash
pip install openai requests python-dotenv streamlit plotly pandas
```

**"Failed to create OpenAI client" Error**
- Verify your OpenAI API key is correct
- Check that you have GPT-5 access enabled
- Ensure your account has sufficient credits

**"Response received but content extraction failed" Warning**
- This indicates a response parsing issue
- The raw response will be displayed for debugging
- Try different model variants or simpler prompts

**Empty Results or Slow Performance**
- Check your internet connection
- Verify API rate limits aren't exceeded
- Clear test results using the sidebar button

## 💡 Usage Tips

1. **Start with Basic Connectivity** to verify your setup
2. **Use Model Comparison** to understand performance differences
3. **Test Reasoning Effort** with complex problems to see the impact
4. **Experiment with Verbosity** for different response lengths
5. **Export Results** regularly to track performance over time
6. **Clear Results** periodically to maintain performance

## 📝 Example Test Scenarios

### Reasoning Effort Test
Default prompt: "You have 12 balls, one of which is either heavier or lighter than the others. Using a balance scale exactly 3 times, how can you identify the odd ball and determine if it's heavier or lighter?"

### Verbosity Test
Default prompt: "Explain how HTTPS encryption works and why it's important for web security"

### Coding Tests
- **Frontend Generation**: React component with TypeScript and Tailwind CSS
- **Bug Fixing**: Python function debugging
- **SQL Generation**: Complex database queries
- **Algorithm Implementation**: Dynamic programming solutions

## 🔄 Session Management

- Test results persist during your browser session
- Use "Clear All Results" in the sidebar to reset
- Session data is not saved between browser sessions
- Export important results before closing the application

## 📄 Notes

- This is a single-file Streamlit application
- All functionality is contained in `gpt-5-testing.py`
- No external configuration files are required
- API keys are handled securely without persistent storage

---

**Single-file GPT-5 testing solution for comprehensive API evaluation and performance analysis.**
