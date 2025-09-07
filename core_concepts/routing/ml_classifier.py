import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def _create_and_save_ml_classifier(model_path: str):
        """Create a new ML classifier and save it to pickle file"""
        # Expanded training data for better model performance
        training_data = [
            # Order Status examples
            ("Where is my order?", "order_status"),
            ("Track my package", "order_status"),
            ("When will my order arrive?", "order_status"),
            ("Order delivery status", "order_status"),
            ("Shipment tracking number", "order_status"),
            ("Expected delivery date", "order_status"),
            ("Package location", "order_status"),
            ("Shipping updates", "order_status"),
            
            # Product Info examples
            ("What are the specs of this product?", "product_info"),
            ("Is this item available?", "product_info"),
            ("Product features and price", "product_info"),
            ("Tell me about this product", "product_info"),
            ("Product dimensions and weight", "product_info"),
            ("Compatible accessories", "product_info"),
            ("Product reviews", "product_info"),
            ("Item specifications", "product_info"),
            
            # Technical Support examples
            ("My app is not working", "technical_support"),
            ("How to fix this error?", "technical_support"),
            ("Troubleshooting help needed", "technical_support"),
            ("Technical issue with software", "technical_support"),
            ("Login problems", "technical_support"),
            ("System error message", "technical_support"),
            ("Bug report", "technical_support"),
            ("Software malfunction", "technical_support"),
            
            # Billing examples
            ("Refund my payment", "billing"),
            ("Billing question", "billing"),
            ("Wrong charge on my card", "billing"),
            ("Invoice inquiry", "billing"),
            ("Payment method update", "billing"),
            ("Subscription cancellation", "billing"),
            ("Credit card issue", "billing"),
            ("Payment failed", "billing"),
            
            # General examples
            ("General question", "general"),
            ("Can you help me?", "general"),
            ("Other inquiry", "general"),
            ("Company information", "general"),
            ("Contact details", "general"),
            ("Store hours", "general"),
            ("About your service", "general"),
            ("General assistance", "general")
        ]
        
        X = [item[0] for item in training_data]
        y = [item[1] for item in training_data]
        
        # Create pipeline with TF-IDF and Logistic Regression
        classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        classifier.fit(X, y)
        
        # Save the trained model to pickle file
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        print(f"Classifier trained and saved to {model_path}")
        return classifier


_create_and_save_ml_classifier("core_concepts/routing/customer_service_classifier.pkl")