import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, List, Any
import anthropic

# Initialize Anthropic client
def get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or st.sidebar.text_input(
        "Anthropic API Key", 
        type="password",
        help="Enter your Anthropic API key"
    )
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    return None

def load_fashion_data(uploaded_file):
    """Load ASOS fashion dataset from uploaded CSV."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def analyze_csv_structure(client, df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze CSV structure once and cache the column understanding."""
    
    # Get column names and sample data
    columns = df.columns.tolist()
    sample_row = df.iloc[0].to_dict() if not df.empty else {}
    
    # Clean sample data for readability (truncate long values)
    cleaned_sample = {}
    for key, value in sample_row.items():
        if pd.isna(value):
            cleaned_sample[key] = "null"
        else:
            str_val = str(value)
            cleaned_sample[key] = str_val[:50] + "..." if len(str_val) > 50 else str_val
    
    prompt = f"""
    Analyze this fashion dataset structure and create a mapping for intelligent search.
    
    Columns: {columns}
    Sample data: {cleaned_sample}
    
    Please analyze and return a JSON object with:
    - "column_mapping": object mapping standard fields to actual column names:
      - "name_columns": columns containing product names/titles
      - "price_columns": columns with pricing info  
      - "brand_columns": columns with brand information
      - "color_columns": columns with color information
      - "category_columns": columns with product categories/types
      - "description_columns": columns with detailed descriptions
      - "size_columns": columns with size information
      - "url_columns": columns with product URLs
      - "image_columns": columns with image URLs
      - "other_searchable": other columns that could be useful for search
    - "data_insights": brief analysis of what kind of fashion data this contains
    
    Focus on identifying which actual column names correspond to these standard fields.
    Return only valid JSON.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        json_str = response.content[0].text
        structure_analysis = json.loads(json_str)
        return structure_analysis
        
    except Exception as e:
        st.error(f"Error analyzing CSV structure: {str(e)}")
        return {}

def analyze_fashion_query(client, query: str, column_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Use Claude to analyze fashion query with cached column understanding."""
    
    prompt = f"""
    You are a fashion search assistant. Analyze this user query and extract search criteria.
    
    User query: "{query}"
    
    Available dataset has these column types:
    {json.dumps(column_mapping.get('column_mapping', {}), indent=2)}
    
    Dataset contains: {column_mapping.get('data_insights', 'fashion products')}
    
    Extract search criteria and return a JSON object with these fields:
    - "keywords": array of relevant keywords to search in product names/descriptions
    - "colors": array of color keywords if mentioned
    - "categories": array of clothing categories/types (dress, shirt, pants, etc.)
    - "occasions": array of occasions if mentioned (casual, formal, party, work, etc.) 
    - "styles": array of style descriptors (vintage, modern, boho, edgy, etc.)
    - "price_range": object with "min" and "max" if price mentioned
    - "brands": array of brand names if mentioned
    - "sizes": array of sizes if mentioned
    - "gender": "men", "women", or "unisex" if determinable
    - "target_columns": suggest which specific dataset columns to search based on the query
    - "interpretation": brief explanation of what the user is looking for
    
    For "revenge dress" - this typically means a stunning, confidence-boosting dress worn after a breakup, usually black, elegant, and attention-grabbing.
    
    Return only valid JSON.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        json_str = response.content[0].text
        criteria = json.loads(json_str)
        return criteria
        
    except Exception as e:
        st.error(f"Error analyzing query: {str(e)}")
        return {}

def filter_fashion_data(df: pd.DataFrame, criteria: Dict[str, Any], column_mapping: Dict[str, Any]) -> pd.DataFrame:
    """Filter fashion dataset using cached column mapping for more precise filtering."""
    filtered_df = df.copy()
    
    # Get mapped columns
    col_map = column_mapping.get('column_mapping', {})
    
    # Get all searchable text columns
    searchable_columns = []
    for col_type, cols in col_map.items():
        if isinstance(cols, list):
            searchable_columns.extend(cols)
        elif isinstance(cols, str):
            searchable_columns.append(cols)
    
    # Remove duplicates and ensure columns exist in dataframe
    searchable_columns = list(set([col for col in searchable_columns if col in df.columns]))
    
    # Apply keyword filtering across relevant columns
    if criteria.get('keywords'):
        keyword_mask = pd.Series([False] * len(df))
        target_cols = criteria.get('target_columns', searchable_columns)
        valid_target_cols = [col for col in target_cols if col in df.columns]
        search_cols = valid_target_cols if valid_target_cols else searchable_columns
        
        for keyword in criteria['keywords']:
            for col in search_cols:
                keyword_mask |= df[col].astype(str).str.contains(keyword, case=False, na=False)
        filtered_df = filtered_df[keyword_mask]
    
    # Apply color filtering using mapped color columns
    if criteria.get('colors'):
        color_columns = col_map.get('color_columns', [])
        if not color_columns:  # Fallback to all searchable columns
            color_columns = searchable_columns
            
        color_mask = pd.Series([False] * len(filtered_df))
        for color in criteria['colors']:
            for col in color_columns:
                if col in filtered_df.columns:
                    color_mask |= filtered_df[col].astype(str).str.contains(color, case=False, na=False)
        if color_mask.any():  # Only apply if we found color matches
            filtered_df = filtered_df[color_mask]
    
    # Apply category filtering using mapped category columns
    if criteria.get('categories'):
        category_columns = col_map.get('category_columns', []) + col_map.get('name_columns', [])
        if not category_columns:
            category_columns = searchable_columns
            
        category_mask = pd.Series([False] * len(filtered_df))
        for category in criteria['categories']:
            for col in category_columns:
                if col in filtered_df.columns:
                    category_mask |= filtered_df[col].astype(str).str.contains(category, case=False, na=False)
        if category_mask.any():
            filtered_df = filtered_df[category_mask]
    
    # Apply brand filtering using mapped brand columns
    if criteria.get('brands'):
        brand_columns = col_map.get('brand_columns', [])
        if brand_columns:
            brand_mask = pd.Series([False] * len(filtered_df))
            for brand in criteria['brands']:
                for col in brand_columns:
                    if col in filtered_df.columns:
                        brand_mask |= filtered_df[col].astype(str).str.contains(brand, case=False, na=False)
            if brand_mask.any():
                filtered_df = filtered_df[brand_mask]
    
    # Apply price filtering using mapped price columns
    if criteria.get('price_range'):
        price_columns = col_map.get('price_columns', [])
        if price_columns:
            for price_col in price_columns:
                if price_col in filtered_df.columns:
                    try:
                        # Try to convert price column to numeric
                        price_series = pd.to_numeric(filtered_df[price_col], errors='coerce')
                        if 'min' in criteria['price_range']:
                            filtered_df = filtered_df[price_series >= criteria['price_range']['min']]
                        if 'max' in criteria['price_range']:
                            filtered_df = filtered_df[price_series <= criteria['price_range']['max']]
                        break  # Use first valid price column
                    except:
                        continue
    
    return filtered_df

def display_products(df: pd.DataFrame, criteria: Dict[str, Any], column_mapping: Dict[str, Any]):
    """Display filtered products using cached column mapping for better presentation."""
    
    if df.empty:
        st.warning("No products found matching your criteria. Try a different search!")
        return
    
    st.success(f"Found {len(df)} matching products")
    
    # Show Claude's interpretation
    if criteria.get('interpretation'):
        st.info(f"**Search interpretation:** {criteria['interpretation']}")
    
    # Display extracted criteria
    with st.expander("🎯 Extracted Search Criteria"):
        criteria_display = {k: v for k, v in criteria.items() if v and k != 'interpretation'}
        st.json(criteria_display)
    
    # Get display columns using cached mapping
    col_map = column_mapping.get('column_mapping', {})
    display_cols = []
    
    # Use mapped columns for better display
    name_cols = col_map.get('name_columns', [])
    price_cols = col_map.get('price_columns', [])
    brand_cols = col_map.get('brand_columns', [])
    color_cols = col_map.get('color_columns', [])
    url_cols = col_map.get('url_columns', [])
    
    # Get first available column of each type
    name_col = next((col for col in name_cols if col in df.columns), None)
    price_col = next((col for col in price_cols if col in df.columns), None)
    brand_col = next((col for col in brand_cols if col in df.columns), None)
    color_col = next((col for col in color_cols if col in df.columns), None)
    url_col = next((col for col in url_cols if col in df.columns), None)
    
    # Build display columns list
    for col in [name_col, price_col, brand_col, color_col, url_col]:
        if col:
            display_cols.append(col)
    
    # Add other important columns
    for col_list in col_map.values():
        if isinstance(col_list, list):
            for col in col_list:
                if col in df.columns and col not in display_cols:
                    display_cols.append(col)
    
    # Limit results for display
    display_df = df.head(20)
    
    # Show results in grid format
    cols = st.columns(3)
    for idx, (_, row) in enumerate(display_df.iterrows()):
        with cols[idx % 3]:
            with st.container():
                st.markdown("---")
                
                # Product name/title
                if name_col and pd.notna(row[name_col]):
                    st.subheader(str(row[name_col])[:50] + "..." if len(str(row[name_col])) > 50 else str(row[name_col]))
                
                # Price
                if price_col and pd.notna(row[price_col]):
                    st.markdown(f"**Price:** {row[price_col]}")
                
                # Brand
                if brand_col and pd.notna(row[brand_col]):
                    st.markdown(f"**Brand:** {row[brand_col]}")
                
                # Color
                if color_col and pd.notna(row[color_col]):
                    st.markdown(f"**Color:** {row[color_col]}")
                
                # URL
                if url_col and pd.notna(row[url_col]):
                    st.markdown(f"[View Product]({row[url_col]})")
    
    # Show CSV structure analysis
    with st.expander("📊 Dataset Analysis"):
        if column_mapping.get('data_insights'):
            st.write(f"**Data insights:** {column_mapping['data_insights']}")
        
        st.write("**Column mapping:**")
        st.json(column_mapping.get('column_mapping', {}))
    
    # Show full results table
    st.subheader("📊 All Results")
    st.dataframe(display_df[display_cols] if display_cols else display_df, use_container_width=True)
    
    # Download option
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="fashion_search_results.csv",
        mime="text/csv"
    )

def main():
    st.set_page_config(
        page_title="Fashion Search Assistant", 
        page_icon="👗",
        layout="wide"
    )
    
    st.title("👗 AI Fashion Search Assistant")
    st.write("Upload your ASOS fashion dataset and search using natural language!")
    
    # Get Anthropic client
    client = get_anthropic_client()
    
    if not client:
        st.warning("Please provide your Anthropic API key to use this app.")
        st.info("You can get an API key from: https://console.anthropic.com/")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload ASOS Fashion Dataset (CSV)", 
        type=['csv'],
        help="Upload your ASOS fashion dataset in CSV format"
    )
    
    if uploaded_file is not None:
        # Load data
        df = load_fashion_data(uploaded_file)
        
        if df is not None:
            st.success(f"Loaded {len(df)} products from dataset")
            
            # Check if we need to analyze CSV structure (do this once per upload)
            if 'csv_structure' not in st.session_state or st.session_state.get('last_uploaded_file') != uploaded_file.name:
                with st.spinner("Analyzing CSV structure..."):
                    try:
                        csv_structure = analyze_csv_structure(client, df)
                        if csv_structure:
                            st.session_state['csv_structure'] = csv_structure
                            st.session_state['last_uploaded_file'] = uploaded_file.name
                            st.success("CSV structure analyzed and cached!")
                        else:
                            st.error("Failed to analyze CSV structure. Using fallback method.")
                            st.session_state['csv_structure'] = {'column_mapping': {}, 'data_insights': 'Fashion dataset'}
                    except Exception as e:
                        st.error(f"Error analyzing CSV: {str(e)}")
                        st.session_state['csv_structure'] = {'column_mapping': {}, 'data_insights': 'Fashion dataset'}
            
            # Show dataset info
            with st.expander("📋 Dataset Info"):
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                st.write("**Sample data:**")
                st.dataframe(df.head(3))
                
                # Show cached structure analysis
                if 'csv_structure' in st.session_state:
                    st.write("**AI Analysis:**")
                    st.write(st.session_state['csv_structure'].get('data_insights', 'No insights available'))
            
            # Search interface
            col1, col2 = st.columns([4, 1])
            
            with col1:
                search_query = st.text_input(
                    "What are you looking for?", 
                    placeholder="e.g., revenge dress, cozy winter sweater, formal black shoes, boho summer dress",
                    help="Describe what you're looking for in natural language!"
                )
            
            with col2:
                search_button = st.button("🔍 Search", type="primary")
            
            # Example queries
            st.markdown("**Try these examples:**")
            example_cols = st.columns(4)
            examples = [
                "revenge dress", 
                "cozy winter sweater under $50",
                "formal black dress for work", 
                "vintage denim jacket"
            ]
            
            for i, example in enumerate(examples):
                if example_cols[i].button(f"💡 {example}", key=f"example_{i}"):
                    search_query = example
                    search_button = True
            
            # Search execution
            if search_button and search_query:
                # Ensure we have cached structure
                if 'csv_structure' not in st.session_state:
                    st.error("CSV structure not analyzed. Please refresh and try again.")
                    return
                    
                with st.spinner(f"Searching for: '{search_query}'..."):
                    try:
                        # Analyze query with cached column mapping (cheap API call)
                        criteria = analyze_fashion_query(client, search_query, st.session_state['csv_structure'])
                        
                        if criteria:
                            # Filter data based on criteria and cached mapping
                            filtered_df = filter_fashion_data(df, criteria, st.session_state['csv_structure'])
                            
                            # Display results with cached mapping
                            display_products(filtered_df, criteria, st.session_state['csv_structure'])
                        else:
                            st.error("Could not analyze your search query. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **Instructions:**
        1. Get an Anthropic API key from https://console.anthropic.com/
        2. Enter your API key in the sidebar or secrets
        3. Upload your ASOS fashion dataset CSV file
        4. Enter natural language fashion queries
        5. View AI-powered search results!
        
        **Example queries:**
        - "revenge dress" - Stunning black dress for confidence
        - "cozy winter sweater under $50" - Warm sweaters within budget
        - "formal work outfit" - Professional attire
        - "boho summer dress" - Bohemian style summer dresses
        - "vintage denim jacket size M" - Specific vintage item with size
        """)
    
    # Technical details
    with st.expander("🔧 How it works"):
        st.markdown("""
        **AI-Powered Search Process:**
        1. **Query Analysis**: Claude analyzes your natural language query
        2. **Criteria Extraction**: Identifies keywords, colors, categories, prices, etc.
        3. **Smart Filtering**: Applies multiple filters to your dataset
        4. **Result Display**: Shows matching products with extracted criteria
        
        **Benefits:**
        - Natural language queries instead of rigid filters
        - Context-aware search (understands "revenge dress", fashion occasions)
        - Multi-criteria filtering across different product attributes
        - Explainable results showing what the AI understood
        """)

if __name__ == "__main__":
    main()
