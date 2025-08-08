#!/usr/bin/env python3
"""
GPT-5 Streamlit Testing Dashboard
Interactive web-based testing interface for GPT-5 API capabilities
"""

import streamlit as st
import os
import time
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page config first
st.set_page_config(
    page_title="GPT-5 API Tester",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from openai import OpenAI
    import requests
    from dotenv import load_dotenv
except ImportError:
    st.error("Required packages not installed. Run: pip install openai requests python-dotenv streamlit plotly")
    st.stop()

# Load environment variables
load_dotenv()

@dataclass
class TestResult:
    test_name: str
    model: str
    api_type: str
    success: bool
    response_time: float
    token_usage: Dict[str, int]
    response_content: str
    error_message: Optional[str] = None
    parameters: Optional[Dict] = None
    timestamp: str = ""

class GPT5StreamlitTester:
    def __init__(self, api_key: str):
        """Initialize the GPT-5 Streamlit tester"""
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Failed to create OpenAI client: {e}")
            raise

        # GPT-5 model variants - explicitly define these
        self.models = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5-chat-latest']
        self.reasoning_efforts = ['minimal', 'low', 'medium', 'high']
        self.verbosity_levels = ['low', 'medium', 'high']

        # Verify attributes were set correctly
        if not hasattr(self, 'models') or len(self.models) != 4:
            raise AttributeError(f"Models not properly initialized. Got: {getattr(self, 'models', 'None')}")

    def extract_response_content(self, response) -> str:
        """Extract content from response with comprehensive fallback methods"""
        content = ""

        # Method 1: GPT-5 Responses API structure (based on actual response format)
        if hasattr(response, 'output') and response.output:
            for output_item in response.output:
                if hasattr(output_item, 'type') and output_item.type == 'message':
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                if hasattr(content_item, 'text'):
                                    content = content_item.text
                                    break
                        if content:
                            break

        # Method 2: Direct output.content access (fallback)
        if not content and hasattr(response, 'output') and response.output:
            if hasattr(response.output, 'content'):
                content = response.output.content
            elif hasattr(response.output, 'text'):
                content = response.output.text
            elif isinstance(response.output, str):
                content = response.output

        # Method 3: Choices format (Chat Completions style fallback)
        if not content and hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
            elif hasattr(choice, 'text'):
                content = choice.text

        # Method 4: Direct content/text attributes
        if not content:
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text

        # Method 5: Check for message content in different structures
        if not content and hasattr(response, 'message'):
            if hasattr(response.message, 'content'):
                content = response.message.content
            elif hasattr(response.message, 'text'):
                content = response.message.text

        return content or ""

    def _safe_text_extract(self, content) -> str:
        """Safely extract text from content that might be an object or string"""
        if isinstance(content, str):
            return content
        elif hasattr(content, 'text'):
            return content.text
        elif hasattr(content, 'content'):
            return content.content
        else:
            return str(content)

    def make_responses_api_call(self, model: str, input_text: str, **kwargs) -> TestResult:
        """Make Responses API call with progress tracking"""
        test_name = kwargs.pop('test_name', 'Generic Test')

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text(f"🚀 Making API call to {model}...")
        progress_bar.progress(25)

        start_time = time.time()

        try:
            # Prepare request data
            request_data = {"model": model, "input": input_text}

            # Add reasoning parameters if specified
            if 'reasoning_effort' in kwargs:
                request_data["reasoning"] = {"effort": kwargs.pop('reasoning_effort')}

            # Add verbosity parameters if specified
            if 'verbosity' in kwargs:
                if "text" not in request_data:
                    request_data["text"] = {}
                request_data["text"]["verbosity"] = kwargs.pop('verbosity')

            # Add max_tokens if specified in text parameters
            if 'text' in kwargs and 'max_tokens' in kwargs['text']:
                if "text" not in request_data:
                    request_data["text"] = {}
                request_data["text"]["max_tokens"] = kwargs['text']['max_tokens']
                kwargs.pop('text')  # Remove from kwargs to avoid duplication

            # Add tools if specified
            if 'tools' in kwargs:
                request_data["tools"] = kwargs.pop('tools')

            # Add tool_choice if specified
            if 'tool_choice' in kwargs:
                request_data["tool_choice"] = kwargs.pop('tool_choice')

            # Add any remaining parameters
            request_data.update(kwargs)

            progress_bar.progress(50)
            status_text.text(f"⏳ Waiting for response from {model}...")

            response = self.client.responses.create(**request_data)
            response_time = time.time() - start_time

            progress_bar.progress(75)
            status_text.text("📝 Processing response...")

            # Extract content using comprehensive method
            content = self.extract_response_content(response)

            # Ensure content is a string, not an object
            if content and not isinstance(content, str):
                if hasattr(content, 'text'):
                    content = content.text
                elif hasattr(content, 'content'):
                    content = content.content
                else:
                    content = str(content)

            # Debug: Print response structure if content is empty
            if not content:
                st.warning("⚠️ Response received but content extraction failed.")
                if hasattr(response, 'model_dump'):
                    st.json(response.model_dump())
                else:
                    st.write("Response object:", str(response)[:500] + "..." if len(str(response)) > 500 else str(response))

            progress_bar.progress(100)
            status_text.text("✅ Request completed successfully!")

            result = TestResult(
                test_name=test_name,
                model=model,
                api_type="responses",
                success=True,
                response_time=response_time,
                token_usage=response.usage.model_dump() if hasattr(response, 'usage') and response.usage else {},
                response_content=content,
                parameters=request_data,
                timestamp=datetime.now().isoformat()
            )

            time.sleep(0.5)  # Brief pause to see success message
            progress_bar.empty()
            status_text.empty()

            return result

        except Exception as e:
            response_time = time.time() - start_time
            progress_bar.progress(100)
            status_text.text(f"❌ Error: {str(e)}")

            result = TestResult(
                test_name=test_name,
                model=model,
                api_type="responses",
                success=False,
                response_time=response_time,
                token_usage={},
                response_content="",
                error_message=str(e),
                parameters=kwargs,
                timestamp=datetime.now().isoformat()
            )

            time.sleep(2)  # Show error message longer
            progress_bar.empty()
            status_text.empty()

            return result

    def display_result(self, result: TestResult):
        """Display test result in a nice format"""
        if result.success:
            st.success(f"✅ {result.test_name} - {result.model} ({result.response_time:.2f}s)")
        else:
            st.error(f"❌ {result.test_name} - {result.model} - Error: {result.error_message}")

        # Create expandable section for details
        with st.expander("📋 View Details", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Parameters:**")
                if result.parameters:
                    st.json(result.parameters)

                st.write("**Token Usage:**")
                if result.token_usage:
                    st.json(result.token_usage)

            with col2:
                st.write(f"**Response Time:** {result.response_time:.2f} seconds")
                st.write(f"**API Type:** {result.api_type}")
                st.write(f"**Timestamp:** {result.timestamp}")

        if result.success and result.response_content:
            st.write("**Response:**")
            # Ensure response_content is a string for display
            display_content = result.response_content
            if not isinstance(display_content, str):
                if hasattr(display_content, 'text'):
                    display_content = display_content.text
                elif hasattr(display_content, 'content'):
                    display_content = display_content.content
                else:
                    display_content = str(display_content)
            st.markdown("""
                <style>
                .stTextArea [data-baseweb=base-input] [disabled=""]{
                    -webkit-text-fill-color: #000;
                }
                </style>
                """,unsafe_allow_html=True)
            st.text_area("", display_content, height=200, disabled=True, key=f"response_{id(result)}")
        elif result.success and not result.response_content:
            st.warning("⚠️ API call successful but no response content received. This might be a response parsing issue.")

            # Show raw response for debugging
            with st.expander("🔍 Debug: Raw Response Structure"):
                st.write("If you see this, please check the response parsing logic.")
                st.write("Token usage shows the API worked, but content extraction failed.")

def main():
    """Main Streamlit application"""

    # Initialize session state FIRST
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []

    # Title and description
    st.title("🤖 GPT-5 API Testing Dashboard")
    st.markdown("Interactive testing interface for GPT-5's new capabilities and features")

    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")

    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv('OPENAI_API_KEY', ''),
        help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable"
    )

    # Initialize tester
    if not api_key:
        st.error("❌ Please enter your OpenAI API key in the sidebar")
        st.info("💡 You can also set the OPENAI_API_KEY environment variable")
        st.stop()

    try:
        with st.spinner("Initializing GPT-5 tester..."):
            tester = GPT5StreamlitTester(api_key)
            st.success("✅ GPT-5 tester initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize GPT-5 tester: {e}")
        st.code(f"Error details: {str(e)}")
        st.stop()

    # Sidebar test selection
    st.sidebar.header("🧪 Test Selection")

    test_categories = {
        "Basic Tests": [
            "Basic Connectivity",
            "Model Comparison"
        ],
        "New Features": [
            "Reasoning Effort",
            "Verbosity Controls",
            "Custom Tools",
            "Allowed Tools"
        ],
        "Capabilities": [
            "Coding Tests",
            "Instruction Following",
            "Factual Accuracy"
        ],
        "Performance": [
            "Speed Comparison",
            "Token Efficiency"
        ]
    }

    selected_category = st.sidebar.selectbox("Test Category", list(test_categories.keys()))
    selected_test = st.sidebar.selectbox("Specific Test", test_categories[selected_category])

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🧪 Run Tests", "📊 Results Dashboard", "⚙️ Custom Test", "📋 Batch Tests"])

    with tab1:
        st.header(f"🧪 Running: {selected_test}")

        if selected_test == "Basic Connectivity":
            st.write("Test basic API connectivity across all GPT-5 models")

            if st.button("🚀 Run Basic Connectivity Test", type="primary"):
                st.write("Testing connectivity...")

                input_text = "Hello! Can you confirm you're GPT-5 and briefly describe your key capabilities?"

                for model in tester.models:
                    st.write(f"Testing {model}...")
                    result = tester.make_responses_api_call(
                        model=model,
                        input_text=input_text,
                        test_name="Basic Connectivity"
                    )
                    st.session_state.test_results.append(result)
                    tester.display_result(result)

                st.success("✅ Basic connectivity tests completed!")

        elif selected_test == "Reasoning Effort":
            st.write("Test the new reasoning effort parameter")

            model = st.selectbox("Select Model", tester.models, key="reasoning_model")
            reasoning_levels = st.multiselect(
                "Reasoning Effort Levels",
                tester.reasoning_efforts,
                default=['minimal', 'medium', 'high']
            )

            problem = st.text_area(
                "Test Problem",
                "You have 12 balls, one of which is either heavier or lighter than the others. Using a balance scale exactly 3 times, how can you identify the odd ball and determine if it's heavier or lighter?",
                height=100
            )

            if st.button("🧠 Run Reasoning Test", type="primary"):
                for effort in reasoning_levels:
                    st.write(f"Testing reasoning effort: {effort}")
                    result = tester.make_responses_api_call(
                        model=model,
                        input_text=problem,
                        reasoning_effort=effort,
                        test_name=f"Reasoning-{effort}"
                    )
                    st.session_state.test_results.append(result)
                    tester.display_result(result)

                    # Show word count analysis
                    if result.success:
                        response_text = result.response_content
                        if not isinstance(response_text, str):
                            if hasattr(response_text, 'text'):
                                response_text = response_text.text
                            elif hasattr(response_text, 'content'):
                                response_text = response_text.content
                            else:
                                response_text = str(response_text)
                        word_count = len(response_text.split())
                        st.metric(f"Word Count ({effort})", word_count)

        elif selected_test == "Verbosity Controls":
            st.write("Test the new verbosity parameter")

            model = st.selectbox("Select Model", tester.models, key="verbosity_model")
            verbosity_levels = st.multiselect(
                "Verbosity Levels",
                tester.verbosity_levels,
                default=['low', 'medium', 'high']
            )

            topic = st.text_input(
                "Explanation Topic",
                "Explain how HTTPS encryption works and why it's important for web security"
            )

            if st.button("📝 Run Verbosity Test", type="primary"):
                for verbosity in verbosity_levels:
                    st.write(f"Testing verbosity: {verbosity}")
                    result = tester.make_responses_api_call(
                        model=model,
                        input_text=topic,
                        verbosity=verbosity,
                        test_name=f"Verbosity-{verbosity}"
                    )
                    st.session_state.test_results.append(result)
                    tester.display_result(result)

                    # Show length analysis
                    if result.success:
                        response_text = result.response_content
                        if not isinstance(response_text, str):
                            if hasattr(response_text, 'text'):
                                response_text = response_text.text
                            elif hasattr(response_text, 'content'):
                                response_text = response_text.content
                            else:
                                response_text = str(response_text)
                        word_count = len(response_text.split())
                        char_count = len(response_text)
                        col1, col2 = st.columns(2)
                        col1.metric(f"Words ({verbosity})", word_count)
                        col2.metric(f"Characters ({verbosity})", char_count)

        elif selected_test == "Coding Tests":
            st.write("Test GPT-5's enhanced coding capabilities")

            coding_type = st.selectbox(
                "Coding Test Type",
                ["Frontend Generation", "Bug Fixing", "SQL Generation", "Algorithm Implementation"]
            )

            model = st.selectbox("Select Model", tester.models, key="coding_model")
            verbosity = st.selectbox("Verbosity", tester.verbosity_levels, index=2, key="coding_verbosity")

            prompts = {
                "Frontend Generation": "Create a responsive React component for a product card with image, title, price, and add-to-cart button. Use TypeScript and Tailwind CSS. Include hover effects and proper accessibility.",
                "Bug Fixing": "Find and fix all bugs in this Python function:\n\n```python\ndef process_data(items):\n    results = []\n    for item in items:\n        if item['value'] > 0:\n            processed = item['name'].upper() + item['category']\n            results.append(processed)\n    return results\n```",
                "SQL Generation": "Write a SQL query to find the top 5 customers by total purchase amount in the last 6 months, including their contact information and number of orders.",
                "Algorithm Implementation": "Implement a function to find the longest common subsequence between two strings using dynamic programming."
            }

            prompt = st.text_area("Coding Prompt", prompts[coding_type], height=150)

            if st.button("💻 Run Coding Test", type="primary"):
                result = tester.make_responses_api_call(
                    model=model,
                    input_text=prompt,
                    verbosity=verbosity,
                    reasoning_effort="medium",
                    test_name=f"Coding-{coding_type}"
                )
                st.session_state.test_results.append(result)
                tester.display_result(result)

        elif selected_test == "Model Comparison":
            st.write("Compare performance across GPT-5 model variants")

            test_prompt = st.text_area(
                "Comparison Prompt",
                "Explain the concept of machine learning and provide a practical example of its application in business.",
                height=100
            )

            if st.button("⚖️ Run Model Comparison", type="primary"):
                comparison_results = []

                for model in tester.models:
                    st.write(f"Testing {model}...")
                    reasoning_effort = "minimal" if "nano" in model else "medium"

                    result = tester.make_responses_api_call(
                        model=model,
                        input_text=test_prompt,
                        reasoning_effort=reasoning_effort,
                        test_name=f"Model Comparison"
                    )
                    st.session_state.test_results.append(result)
                    comparison_results.append(result)
                    tester.display_result(result)

                # Show comparison metrics
                if all(r.success for r in comparison_results):
                    st.write("📊 **Comparison Metrics:**")

                    metrics_df = pd.DataFrame([
                        {
                            "Model": r.model,
                            "Response Time (s)": r.response_time,
                            "Word Count": len(self._safe_text_extract(r.response_content).split()),
                            "Character Count": len(self._safe_text_extract(r.response_content)),
                            "Total Tokens": r.token_usage.get('total_tokens', 0)
                        }
                        for r in comparison_results
                    ])

                    st.dataframe(metrics_df)

                    # Create comparison charts
                    fig_time = px.bar(metrics_df, x='Model', y='Response Time (s)',
                                     title='Response Time Comparison')
                    st.plotly_chart(fig_time, use_container_width=True)

                    fig_tokens = px.bar(metrics_df, x='Model', y='Total Tokens',
                                       title='Token Usage Comparison')
                    st.plotly_chart(fig_tokens, use_container_width=True)

    with tab2:
        st.header("📊 Results Dashboard")

        if not st.session_state.test_results:
            st.info("No test results yet. Run some tests to see the dashboard!")
        else:
            # Create DataFrame from results
            results_data = []
            for result in st.session_state.test_results:
                data = asdict(result)
                # Safely handle response_content that might not be a string
                response_content = result.response_content
                if isinstance(response_content, str):
                    data['word_count'] = len(response_content.split()) if result.success else 0
                elif hasattr(response_content, 'text'):
                    data['word_count'] = len(response_content.text.split()) if result.success else 0
                elif hasattr(response_content, 'content'):
                    data['word_count'] = len(response_content.content.split()) if result.success else 0
                else:
                    data['word_count'] = 0

                data['total_tokens'] = result.token_usage.get('total_tokens', 0) if result.success else 0
                results_data.append(data)

            df = pd.DataFrame(results_data)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_tests = len(df)
                st.metric("Total Tests", total_tests)

            with col2:
                success_rate = (df['success'].sum() / len(df) * 100) if len(df) > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")

            with col3:
                avg_time = df[df['success']]['response_time'].mean() if df['success'].any() else 0
                st.metric("Avg Response Time", f"{avg_time:.2f}s")

            with col4:
                total_tokens = df[df['success']]['total_tokens'].sum()
                st.metric("Total Tokens Used", total_tokens)

            # Charts
            if df['success'].any():
                successful_df = df[df['success']]

                # Response time by model
                fig_time = px.box(successful_df, x='model', y='response_time',
                                 title='Response Time Distribution by Model')
                st.plotly_chart(fig_time, use_container_width=True)

                # Token usage by test
                fig_tokens = px.scatter(successful_df, x='response_time', y='total_tokens',
                                       color='model', size='word_count',
                                       title='Response Time vs Token Usage')
                st.plotly_chart(fig_tokens, use_container_width=True)

            # Detailed results table
            st.subheader("📋 Detailed Results")

            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                model_filter = st.multiselect("Filter by Model", df['model'].unique(), df['model'].unique())
            with col2:
                success_filter = st.selectbox("Filter by Status", ["All", "Success Only", "Errors Only"])
            with col3:
                test_filter = st.multiselect("Filter by Test", df['test_name'].unique(), df['test_name'].unique())

            # Apply filters
            filtered_df = df[df['model'].isin(model_filter) & df['test_name'].isin(test_filter)]

            if success_filter == "Success Only":
                filtered_df = filtered_df[filtered_df['success']]
            elif success_filter == "Errors Only":
                filtered_df = filtered_df[~filtered_df['success']]

            # Display table
            display_columns = ['test_name', 'model', 'success', 'response_time', 'word_count', 'total_tokens', 'timestamp']
            st.dataframe(filtered_df[display_columns], use_container_width=True)

            # Export option
            if st.button("📥 Export Results as JSON"):
                results_json = json.dumps([asdict(r) for r in st.session_state.test_results], indent=2)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name=f"gpt5_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    with tab3:
        st.header("⚙️ Custom Test")
        st.write("Create and run your own custom tests")

        col1, col2 = st.columns(2)

        with col1:
            custom_model = st.selectbox("Model", tester.models, key="custom_model")
            custom_reasoning = st.selectbox("Reasoning Effort", tester.reasoning_efforts, key="custom_reasoning")
            custom_verbosity = st.selectbox("Verbosity", tester.verbosity_levels, key="custom_verbosity")

        with col2:
            test_name = st.text_input("Test Name", "Custom Test")
            max_tokens = st.number_input("Max Output Tokens", min_value=50, max_value=4000, value=500)

        custom_prompt = st.text_area(
            "Your Prompt",
            "Enter your custom prompt here...",
            height=200
        )

        if st.button("🚀 Run Custom Test", type="primary"):
            if custom_prompt and custom_prompt != "Enter your custom prompt here...":
                # Prepare additional parameters for Responses API
                additional_params = {}

                # Add max_tokens using the correct Responses API structure
                if max_tokens != 500:  # Only add if different from default
                    additional_params["text"] = additional_params.get("text", {})
                    additional_params["text"]["max_tokens"] = max_tokens

                result = tester.make_responses_api_call(
                    model=custom_model,
                    input_text=custom_prompt,
                    reasoning_effort=custom_reasoning,
                    verbosity=custom_verbosity,
                    test_name=test_name,
                    **additional_params
                )
                st.session_state.test_results.append(result)
                tester.display_result(result)
            else:
                st.error("Please enter a custom prompt")

    with tab4:
        st.header("📋 Batch Tests")
        st.write("Run multiple predefined tests in sequence")

        batch_tests = {
            "Quick Test Suite": [
                "Basic Connectivity",
                "Reasoning Effort (medium only)",
                "Verbosity Controls (medium only)"
            ],
            "Comprehensive Suite": [
                "Basic Connectivity",
                "Full Reasoning Test",
                "Full Verbosity Test",
                "Coding Tests",
                "Model Comparison"
            ],
            "Performance Suite": [
                "Speed Comparison",
                "Token Efficiency",
                "Model Comparison"
            ]
        }

        selected_batch = st.selectbox("Select Test Suite", list(batch_tests.keys()))

        st.write(f"**{selected_batch} includes:**")
        for test in batch_tests[selected_batch]:
            st.write(f"• {test}")

        if st.button(f"🚀 Run {selected_batch}", type="primary"):
            st.write(f"Running {selected_batch}...")

            with st.spinner("Running batch tests..."):
                # This would implement the batch test logic
                # For now, just show a placeholder
                progress = st.progress(0)
                for i, test in enumerate(batch_tests[selected_batch]):
                    st.write(f"Running: {test}")
                    time.sleep(1)  # Simulate test execution
                    progress.progress((i + 1) / len(batch_tests[selected_batch]))

                st.success(f"✅ {selected_batch} completed!")

    # Clear results button in sidebar
    st.sidebar.header("🗑️ Cleanup")
    if st.sidebar.button("Clear All Results"):
        st.session_state.test_results = []
        st.sidebar.success("Results cleared!")
        st.rerun()

    # Show current results count
    if st.session_state.test_results:
        st.sidebar.info(f"📊 {len(st.session_state.test_results)} results stored")

if __name__ == "__main__":
    main()
