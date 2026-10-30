import os
import re
import json

class FinancialNLPParser:
    def __init__(self):
        # Set up regex-based entity patterns for robust fallback (no spacy download required)
        self.ticker_pattern = re.compile(r'\b[A-Z]{2,5}\b')
        self.monetary_pattern = re.compile(r'\$\d+(?:\.\d+)?\s*(?:billion|million|B|M|percent|%)?')
        self.executive_names = {
            'Tim Cook', 'Luca Maestri', 'Elon Musk', 'Satya Nadella', 'Sundar Pichai',
            'Andy Jassy', 'Jensen Huang', 'Mark Zuckerberg', 'Jamie Dimon', 'CFO', 'CEO'
        }
        self.company_names = {
            'Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla', 'Nvidia', 'Meta',
            'Netflix', 'Berkshire Hathaway', 'JPMorgan'
        }

    def extract_entities(self, text):
        entities = {
            'TICKERS': [],
            'COMPANIES': [],
            'EXECUTIVES': [],
            'METRICS': []
        }
        
        # Extract tickers (excluding common words in all caps)
        exclude_caps = {'SEC', 'CEO', 'CFO', 'GAAP', 'USD', 'USA', 'EPS', 'Q1', 'Q2', 'Q3', 'Q4', 'A', 'I', 'IV'}
        raw_tickers = self.ticker_pattern.findall(text)
        entities['TICKERS'] = list(set([t for t in raw_tickers if t not in exclude_caps]))
        
        # Extract company names
        for company in self.company_names:
            if re.search(r'\b' + re.escape(company) + r'\b', text, re.IGNORECASE):
                entities['COMPANIES'].append(company)
                
        # Extract executives/persons
        for exec_name in self.executive_names:
            if re.search(r'\b' + re.escape(exec_name) + r'\b', text, re.IGNORECASE):
                entities['EXECUTIVES'].append(exec_name)
                
        # Extract monetary values & percentages
        raw_metrics = self.monetary_pattern.findall(text)
        entities['METRICS'] = list(set(raw_metrics))
        
        return entities

    def segment_transcript(self, transcript_text):
        # Segments a transcripts into individual sentences or clauses
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', transcript_text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

def simulate_transcript():
    return """
    Microsoft Corp. (MSFT) announced its Q3 earnings call today. CEO Satya Nadella started the call by highlighting 
    strong demand for Azure cloud services, stating that cloud margins grew by 18 percent this quarter. 
    Our AI infrastructure investments are paying off, and Azure revenue exceeded $25.3 billion.
    However, CFO Amy Hood warned that supply chain constraints could negatively impact hardware shipments.
    Xbox hardware sales fell by 6 percent, resulting in minor margin pressure.
    Overall, Microsoft expects to beat guidance for the next fiscal year.
    Meanwhile, competitors like Apple and Google are also expanding their cloud footprints.
    """

def main():
    print("Initializing NLP Contextual Extraction Pipeline...")
    
    # Load transcript
    text = simulate_transcript()
    print("\n--- Processing Raw Transcript ---")
    print(text.strip())
    
    # Run pipeline
    parser = FinancialNLPParser()
    
    # 1. Segment document
    segments = parser.segment_transcript(text)
    print(f"\nDocument segmented into {len(segments)} sentences.")
    
    # 2. Extract entities for each segment
    print("\nAnalyzing segment entities & local context:")
    print("-" * 80)
    
    context_store = []
    
    for idx, segment in enumerate(segments):
        entities = parser.extract_entities(segment)
        
        # Find if segment is positive, negative, or neutral based on keywords (simple baseline sentiment logic)
        pos_words = ['strong', 'grow', 'exceed', 'pay off', 'beat']
        neg_words = ['warn', 'constraint', 'negatively', 'fall', 'pressure']
        
        pos_score = sum(1 for w in pos_words if w in segment.lower())
        neg_score = sum(1 for w in neg_words if w in segment.lower())
        
        if pos_score > neg_score:
            segment_sentiment = 'positive'
        elif neg_score > pos_score:
            segment_sentiment = 'negative'
        else:
            segment_sentiment = 'neutral'
            
        # Only log segments with entities to maintain context quality
        if any(entities.values()):
            print(f"Segment #{idx+1}: '{segment}'")
            print(f"  Entities:  {entities}")
            print(f"  Sentiment: {segment_sentiment.upper()}")
            print("-" * 80)
            
            context_store.append({
                'segment_id': idx + 1,
                'text': segment,
                'entities': entities,
                'sentiment': segment_sentiment
            })
            
    # Save the Context Store as a JSON artifact
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(CURRENT_DIR, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    out_file = os.path.join(outputs_dir, 'context_store.json')
    with open(out_file, 'w') as f:
        json.dump(context_store, f, indent=4)
        
    print(f"\nSaved context-aware entities to '{out_file}'.")

if __name__ == '__main__':
    main()
