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
    """Load fashion dataset from uploaded CSV."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def create_fallback_mapping(df: pd.DataFrame) -> Dict[str, Any]:
    """Create basic column mapping when LLM analysis fails."""
    fallback_mapping = {
        'column_mapping': {
            'name_columns': [col for col in df.columns if any(word in col.lower() for word in ['name', 'title', 'product'])],
            'price_columns': [col for col in df.columns if 'price' in col.lower()],
            'brand_columns': [col for col in df.columns if 'brand' in col.lower()],
            'color_columns': [col for col in df.columns if 'color' in col.lower() or 'colour' in col.lower()],
            'category_columns': [col for col in df.columns if any(word in col.lower() for word in ['category', 'type', 'section'])],
            'description_columns': [col for col in df.columns if any(word in col.lower() for word in ['description', 'detail'])],
            'size_columns': [col for col in df.columns if 'size' in col.lower()],
            'url_columns': [col for col in df.columns if 'url' in col.lower() or 'link' in col.lower()],
            'image_columns': [col for col in df.columns if 'image' in col.lower() or 'img' in col.lower()],
            'other_searchable': []
        },
        'data_insights': f'Fashion dataset with {len(df)} products - automatic analysis'
    }
    return fallback_mapping

def clean_json_response(response_text: str) -> str:
    """Clean LLM response to extract valid JSON."""
    # Remove markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    
    # Remove any leading/trailing whitespace
    response_text = response_text.strip()
    
    # Try to find JSON object boundaries
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        response_text = response_text[start_idx:end_idx + 1]
    
    return response_text

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
    IMPORTANT: Return ONLY valid JSON without any markdown formatting, explanations, or code blocks.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Clean and parse JSON response
        json_str = clean_json_response(response.content[0].text)
        
        # Debug: show raw response if needed
        if st.session_state.get('debug_mode', False):
            st.write("Raw LLM response:", response.content[0].text)
            st.write("Cleaned JSON:", json_str)
        
        structure_analysis = json.loads(json_str)
        return structure_analysis
        
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing failed: {str(e)[:100]}... Using fallback method.")
        return create_fallback_mapping(df)
    except Exception as e:
        st.warning(f"CSV analysis failed: {str(e)[:100]}... Using fallback method.")
        return create_fallback_mapping(df)

def analyze_fashion_query(client, query: str, column_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Use Claude to analyze fashion query with cached column understanding and generate expanded search terms."""
    
    prompt = f"""
    You are a fashion search assistant. Analyze this user query and extract comprehensive search criteria with synonym expansion.
    
    User query: "{query}"
    
    Available dataset has these column types:
    {json.dumps(column_mapping.get('column_mapping', {}), indent=2)}
    
    Dataset contains: {column_mapping.get('data_insights', 'fashion products')}
    
    Extract search criteria and return a JSON object with these fields:
    - "keywords": array of relevant keywords to search in product names/descriptions
    - "expanded_keywords": array of synonyms and alternative terms that fashion brands might use
    - "colors": array of color keywords if mentioned
    - "categories": array of clothing categories/types (dress, shirt, pants, etc.)
    - "occasions": array of occasions if mentioned (casual, formal, party, work, etc.) 
    - "styles": array of style descriptors (vintage, modern, boho, edgy, etc.)
    - "expanded_styles": array of alternative style terms (vintage -> retro, heritage, classic, throwback)
    - "price_range": object with "min" and "max" if price mentioned
    - "brands": array of brand names if mentioned
    - "sizes": array of sizes if mentioned
    - "gender": "men", "women", or "unisex" if determinable
    - "target_columns": suggest which specific dataset columns to search based on the query
    - "interpretation": brief explanation of what the user is looking for
    
    When generating expanded keywords, think about how e-commerce sites and fashion retailers would describe these products in their catalogs. Include:
    - Retail category terms and merchandising language (e.g., "lifestyle sneakers", "court classics", "heritage footwear")
    - Brand terminology and marketing descriptors (e.g., "street style", "smart casual", "elevated basics")
    
    For synonym expansion, think about how fashion brands describe items:
    - "vintage" could be: retro, heritage, classic, throwback, timeless
    - "cozy" could be: comfortable, soft, warm, plush, snug
    - "jacket" could be: coat, outerwear, blazer, cardigan, wrap
    - "casual shoes" could be: sneakers, trainers, lifestyle footwear, court shoes, low-tops
    
    For "revenge dress" - this typically means a stunning, confidence-boosting dress worn after a breakup, usually black, elegant, and attention-grabbing.
    
    IMPORTANT: Return ONLY valid JSON without any markdown formatting, explanations, or code blocks.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Clean and parse JSON response
        json_str = clean_json_response(response.content[0].text)
        
        # Debug: show raw response if needed
        if st.session_state.get('debug_mode', False):
            st.write("Query analysis response:", json_str)
        
        criteria = json.loads(json_str)
        return criteria
        
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse search criteria: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"Error analyzing query: {str(e)}")
        return {}

def filter_fashion_data(df: pd.DataFrame, criteria: Dict[str, Any], column_mapping: Dict[str, Any]) -> pd.DataFrame:
    """Filter fashion dataset using cached column mapping with expanded keyword search for broader results."""
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
    
    # Apply keyword filtering with expanded terms for broader results
    if criteria.get('keywords') or criteria.get('expanded_keywords'):
        keyword_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        target_cols = criteria.get('target_columns', searchable_columns)
        valid_target_cols = [col for col in target_cols if col in filtered_df.columns]
        search_cols = valid_target_cols if valid_target_cols else searchable_columns
        
        # Combine original and expanded keywords
        all_keywords = criteria.get('keywords', []) + criteria.get('expanded_keywords', [])
        
        for keyword in all_keywords:
            for col in search_cols:
                keyword_mask |= filtered_df[col].astype(str).str.contains(keyword, case=False, na=False)
        filtered_df = filtered_df[keyword_mask]
    
    # Apply style filtering with expanded terms
    if criteria.get('styles') or criteria.get('expanded_styles'):
        style_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        style_columns = col_map.get('category_columns', []) + col_map.get('name_columns', []) + col_map.get('description_columns', [])
        if not style_columns:
            style_columns = searchable_columns
        
        # Combine original and expanded styles
        all_styles = criteria.get('styles', []) + criteria.get('expanded_styles', [])
        
        for style in all_styles:
            for col in style_columns:
                if col in filtered_df.columns:
                    style_mask |= filtered_df[col].astype(str).str.contains(style, case=False, na=False)
        if style_mask.any():
            filtered_df = filtered_df[style_mask]
    
    # Apply color filtering using mapped color columns
    if criteria.get('colors'):
        color_columns = col_map.get('color_columns', [])
        if not color_columns:  # Fallback to all searchable columns
            color_columns = searchable_columns
            
        color_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
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
            
        category_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
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
            brand_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
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
                        # Clean price data - remove currency symbols and convert to numeric
                        price_series = filtered_df[price_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                        price_series = pd.to_numeric(price_series, errors='coerce')
                        
                        if 'min' in criteria['price_range']:
                            filtered_df = filtered_df[price_series >= criteria['price_range']['min']]
                        if 'max' in criteria['price_range']:
                            filtered_df = filtered_df[price_series <= criteria['price_range']['max']]
                        break  # Use first valid price column
                    except Exception as e:
                        continue
    
    return filtered_df

def rank_products_with_llm(client, df: pd.DataFrame, original_query: str, column_mapping: Dict[str, Any]) -> pd.DataFrame:
    """Use LLM to intelligently rank and select top 10 products from broader search results."""
    
    if len(df) <= 10:
        return df  # No need to rank if we have 10 or fewer results
    
    # Get key columns for ranking
    col_map = column_mapping.get('column_mapping', {})
    name_col = next((col for col in col_map.get('name_columns', []) if col in df.columns), None)
    price_col = next((col for col in col_map.get('price_columns', []) if col in df.columns), None)
    brand_col = next((col for col in col_map.get('brand_columns', []) if col in df.columns), None)
    desc_col = next((col for col in col_map.get('description_columns', []) if col in df.columns), None)
    
    # Prepare product data for LLM - limit to top 20 for cost control
    products_for_ranking = df.head(20)
    products_list = []
    
    for idx, (_, row) in enumerate(products_for_ranking.iterrows()):
        product_data = {
            "id": idx,
            "name": str(row[name_col])[:100] if name_col and pd.notna(row[name_col]) else "Unknown",
            "price": str(row[price_col]) if price_col and pd.notna(row[price_col]) else "N/A",
            "brand": str(row[brand_col]) if brand_col and pd.notna(row[brand_col]) else "N/A"
        }
        
        # Add description if available (truncated)
        if desc_col and pd.notna(row[desc_col]):
            product_data["description"] = str(row[desc_col])[:150] + "..." if len(str(row[desc_col])) > 150 else str(row[desc_col])
        
        products_list.append(product_data)
    
    # Create ranking prompt
    prompt = f"""
    You are a professional stylist with years of experience in fashion media, personal styling, and understanding what works in real-world settings. A client searched for: "{original_query}"

    Here are the available options from their wardrobe/shopping list:
    {json.dumps(products_list, indent=1)}

    As their stylist, rank these items by what you would actually recommend for this specific request. Draw on your knowledge of:

    STYLIST EXPERTISE:
    - What fashion editors feature in magazines for similar occasions
    - What influencers and style icons actually wear in these situations  
    - How different pieces photograph and present in social settings
    - Which items have lasting appeal vs fleeting trends
    - Price-to-style ratio and investment piece value

    REAL-WORLD STYLING:
    - Consider comfort and confidence - the best outfit is one they feel great in
    - Think about versatility - can this work for multiple similar occasions?
    - Account for current fashion climate and what's considered stylish now
    - Balance aspiration with practicality for the stated use case
    - Consider visual appeal and how pieces present in photos, as this affects confidence and satisfaction

    YOUR RECOMMENDATION PROCESS:
    - What would you personally put together for a client with this request?
    - Which pieces have that "effortlessly chic" quality stylists look for?
    - What creates the right impression for the specific occasion mentioned?
    - How do these items fit into current fashion narratives and cultural moments?

    Rank as if you're curating a personalized selection for a valued client who trusts your fashion judgment.

    Return a JSON object with:
    - "top_products": array of the top 10 product IDs in order of preference (your top recommendations first)
    - "reasoning": your styling rationale focusing on why these choices work for this specific request

    IMPORTANT: Return ONLY valid JSON without any markdown formatting, explanations, or code blocks.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Clean and parse JSON response
        json_str = clean_json_response(response.content[0].text)
        
        if st.session_state.get('debug_mode', False):
            st.write("Ranking response:", json_str)
        
        ranking_result = json.loads(json_str)
        top_product_ids = ranking_result.get('top_products', [])
        
        # Reorder dataframe based on LLM ranking
        if top_product_ids:
            # Get the ranked products
            ranked_indices = [products_for_ranking.index[i] for i in top_product_ids if i < len(products_for_ranking)]
            ranked_df = df.loc[ranked_indices]
            
            # Add ranking metadata
            if 'reasoning' in ranking_result:
                st.session_state['ranking_reasoning'] = ranking_result['reasoning']
            
            return ranked_df
        else:
            return df.head(10)  # Fallback to first 10 if ranking fails
            
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Ranking failed: {str(e)}")
        return df.head(10)  # Fallback to first 10 if ranking fails

def display_products(df: pd.DataFrame, criteria: Dict[str, Any], column_mapping: Dict[str, Any]):
    """Display filtered products using cached column mapping for better presentation."""
    
    if df.empty:
        st.warning("No products found matching your criteria. Try a different search!")
        return
    
    st.success(f"Found {len(df)} matching products")
    
    # Show Claude's interpretation
    if criteria.get('interpretation'):
        st.info(f"Search interpretation: {criteria['interpretation']}")
    
    # Show ranking reasoning if available
    if 'ranking_reasoning' in st.session_state:
        st.info(f"Ranking reasoning: {st.session_state['ranking_reasoning']}")
    
    # Display extracted criteria
    with st.expander("Search Criteria Extracted"):
        # Show expanded search terms
        if criteria.get('expanded_keywords'):
            st.write("Original keywords:", criteria.get('keywords', []))
            st.write("Expanded keywords:", criteria.get('expanded_keywords', []))
        if criteria.get('expanded_styles'):
            st.write("Original styles:", criteria.get('styles', []))
            st.write("Expanded styles:", criteria.get('expanded_styles', []))
        
        criteria_display = {k: v for k, v in criteria.items() if v and k not in ['interpretation', 'expanded_keywords', 'expanded_styles']}
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
    
    # Show products with improved visual layout
    st.markdown("### Recommended Products")
    
    cols = st.columns(2)
    for idx, (_, row) in enumerate(display_df.iterrows()):
        with cols[idx % 2]:
            with st.container():
                # Product image handling
                image_col = next((col for col in col_map.get('image_columns', []) if col in df.columns), None)
                image_url = row.get(image_col, '') if image_col else ''
                
                if image_url and str(image_url).strip() and str(image_url) != 'N/A':
                    try:
                        # Clean image URL - handle multiple URLs separated by delimiters
                        if '~' in str(image_url):
                            image_url = str(image_url).split('~')[0].strip()
                        elif ',' in str(image_url):
                            image_url = str(image_url).split(',')[0].strip()
                        
                        if str(image_url).startswith('http'):
                            product_name = str(row[name_col])[:40] if name_col and pd.notna(row[name_col]) else "Product"
                            st.image(image_url, width=200, caption=product_name)
                        else:
                            st.write("Image: Invalid URL")
                    except Exception:
                        st.write("Image: Loading error")
                else:
                    st.write("Image: Not available")
                
                # Product name
                if name_col and pd.notna(row[name_col]):
                    product_name = str(row[name_col])
                    display_name = product_name[:50] + "..." if len(product_name) > 50 else product_name
                    st.markdown(f"**{display_name}**")
                
                # Create two columns for details and button
                col_details, col_button = st.columns([2, 1])
                
                with col_details:
                    # Price 
                    if price_col and pd.notna(row[price_col]):
                        st.write(f"Price: **{row[price_col]}**")
                    
                    # Brand
                    if brand_col and pd.notna(row[brand_col]):
                        st.write(f"Brand: {row[brand_col]}")
                    
                    # Color
                    if color_col and pd.notna(row[color_col]):
                        st.write(f"Color: {row[color_col]}")
                    
                    # Size if available
                    size_cols = col_map.get('size_columns', [])
                    size_col = next((col for col in size_cols if col in df.columns), None)
                    if size_col and pd.notna(row[size_col]):
                        st.write(f"Size: {row[size_col]}")
                
                with col_button:
                    # Product URL button
                    if url_col and pd.notna(row[url_col]) and str(row[url_col]) not in ['#', '', 'N/A']:
                        st.link_button("View Product", str(row[url_col]), use_container_width=True)
                
                # Description in expandable section
                desc_cols = col_map.get('description_columns', [])
                desc_col = next((col for col in desc_cols if col in df.columns), None)
                if desc_col and pd.notna(row[desc_col]):
                    description = str(row[desc_col])
                    if description not in ['', 'N/A'] and len(description) > 10:
                        with st.expander("Description"):
                            desc_text = description[:200]
                            if len(description) > 200:
                                desc_text += "..."
                            st.write(desc_text)
                
                st.markdown("---")
    
    # Show CSV structure analysis
    with st.expander("Dataset Analysis"):
        if column_mapping.get('data_insights'):
            st.write(f"Data insights: {column_mapping['data_insights']}")
        
        st.write("Column mapping:")
        st.json(column_mapping.get('column_mapping', {}))
    
    # Show full results table
    st.subheader("All Results")
    st.dataframe(display_df[display_cols] if display_cols else display_df, use_container_width=True)
    
    # Download option
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
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
    
    st.title("Fashion Search Assistant")
    st.write("Upload your fashion dataset and search using natural language!")
    
    # Debug mode toggle
    st.sidebar.checkbox("Debug Mode", key="debug_mode", help="Show raw LLM responses for debugging")
    
    # Get Anthropic client
    client = get_anthropic_client()
    
    if not client:
        st.warning("Please provide your Anthropic API key to use this app.")
        st.info("You can get an API key from: https://console.anthropic.com/")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Fashion Dataset (CSV)", 
        type=['csv'],
        help="Upload your fashion dataset in CSV format"
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
                            
                            if csv_structure == create_fallback_mapping(df):
                                st.info("Using basic column detection (LLM analysis failed)")
                            else:
                                st.success("CSV structure analyzed and cached!")
                        else:
                            st.error("Failed to analyze CSV structure. Using fallback method.")
                            st.session_state['csv_structure'] = create_fallback_mapping(df)
                    except Exception as e:
                        st.error(f"Error analyzing CSV: {str(e)}")
                        st.session_state['csv_structure'] = create_fallback_mapping(df)
            
            # Show dataset info
            with st.expander("Dataset Info"):
                st.write(f"Shape: {df.shape}")
                st.write(f"Columns: {', '.join(df.columns.tolist())}")
                st.write("Sample data:")
                st.dataframe(df.head(3))
                
                # Show cached structure analysis
                if 'csv_structure' in st.session_state:
                    st.write("AI Analysis:")
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
                search_button = st.button("Search", type="primary")
            
            # Example queries
            st.markdown("Try these examples:")
            example_cols = st.columns(4)
            examples = [
                "revenge dress", 
                "cozy winter sweater under $50",
                "formal black dress for work", 
                "vintage denim jacket"
            ]
            
            for i, example in enumerate(examples):
                if example_cols[i].button(f"{example}", key=f"example_{i}"):
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
                        # Step 1: Analyze query with expanded keywords (moderate API call)
                        criteria = analyze_fashion_query(client, search_query, st.session_state['csv_structure'])
                        
                        if criteria:
                            # Step 2: Filter data with expanded search terms for broader results
                            filtered_df = filter_fashion_data(df, criteria, st.session_state['csv_structure'])
                            
                            if len(filtered_df) == 0:
                                st.warning("No products found matching your criteria. Try a different search!")
                                return
                            
                            # Step 3: Use LLM to rank and select top products if we have many results
                            if len(filtered_df) > 10:
                                with st.spinner("Ranking products by relevance..."):
                                    final_df = rank_products_with_llm(client, filtered_df, search_query, st.session_state['csv_structure'])
                            else:
                                final_df = filtered_df
                            
                            # Display results with cached mapping
                            display_products(final_df, criteria, st.session_state['csv_structure'])
                        else:
                            st.error("Could not analyze your search query. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
    
    # Instructions
    with st.expander("How to use"):
        st.markdown("""
        Instructions:
        1. Get an Anthropic API key from https://console.anthropic.com/
        2. Enter your API key in the sidebar or secrets
        3. Upload your fashion dataset CSV file
        4. Enter natural language fashion queries
        5. View AI-powered search results!
        
        Example queries:
        - "revenge dress" - Stunning black dress for confidence
        - "cozy winter sweater under $50" - Warm sweaters within budget
        - "formal work outfit" - Professional attire
        - "boho summer dress" - Bohemian style summer dresses
        - "vintage denim jacket size M" - Specific vintage item with size
        """)
    
    # Technical details
    with st.expander("How it works"):
        st.markdown("""
        AI-Powered Search Process:
        1. Structure Analysis: Claude analyzes your CSV structure once and caches the understanding
        2. Query Analysis: Claude interprets your natural language query using cached context
        3. Synonym Expansion: Generates alternative terms fashion brands might use
        4. Broad Search: Searches with expanded keywords for comprehensive results
        5. Intelligent Ranking: LLM ranks results by relevance to original intent
        6. Result Display: Shows top matches with explanation of AI understanding
        
        Benefits:
        - Natural language queries instead of rigid filters
        - Context-aware search (understands "revenge dress", fashion occasions)
        - Synonym expansion for better coverage
        - Intelligent ranking for most relevant results
        - Cost-effective with session-based caching
        """)

if __name__ == "__main__":
    main()
