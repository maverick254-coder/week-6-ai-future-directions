"""
Ethics in Personalized Medicine AI
Author: Student
Date: July 2025

This module analyzes potential biases in AI-driven personalized medicine
and proposes fairness strategies for equitable healthcare outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class PersonalizedMedicineEthicsAnalyzer:
    """
    Analyzer for identifying and addressing biases in personalized medicine AI systems.
    """
    
    def __init__(self):
        self.model = None
        self.synthetic_data = None
        self.bias_analysis = {}
        
    def generate_synthetic_genomic_data(self, n_samples=10000):
        """
        Generate synthetic genomic and demographic data similar to TCGA.
        
        Args:
            n_samples (int): Number of synthetic patients
            
        Returns:
            pd.DataFrame: Synthetic patient data
        """
        np.random.seed(42)
        
        # Demographics with realistic distributions
        ethnicities = ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Native American', 'Other']
        ethnicity_probs = [0.65, 0.15, 0.08, 0.08, 0.02, 0.02]  # Reflects typical medical dataset bias
        
        genders = ['Male', 'Female']
        gender_probs = [0.52, 0.48]
        
        cancer_types = ['Breast', 'Lung', 'Colorectal', 'Prostate', 'Melanoma', 'Leukemia']
        
        data = []
        
        for i in range(n_samples):
            # Basic demographics
            ethnicity = np.random.choice(ethnicities, p=ethnicity_probs)
            gender = np.random.choice(genders, p=gender_probs)
            age = np.random.normal(60, 15)  # Cancer patients tend to be older
            age = max(18, min(95, age))  # Reasonable age bounds
            
            # Socioeconomic factors
            income_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.4, 0.2])
            insurance_type = np.random.choice(['Public', 'Private', 'Uninsured'], p=[0.5, 0.4, 0.1])
            
            # Geographic factors
            region = np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.4, 0.4, 0.2])
            
            # Cancer characteristics
            cancer_type = np.random.choice(cancer_types)
            stage = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.25, 0.15])
            
            # Synthetic genomic markers (simplified)
            # In reality, these would be complex genomic sequences
            genetic_markers = {
                f'marker_{j}': np.random.choice([0, 1], p=[0.7, 0.3]) 
                for j in range(20)  # 20 simplified genetic markers
            }
            
            # Treatment response (influenced by bias)
            # Simulate historical bias where certain groups had worse outcomes
            base_response_prob = 0.7
            
            # Introduce systematic biases
            if ethnicity in ['African American', 'Hispanic', 'Native American']:
                base_response_prob -= 0.15  # Historical healthcare disparities
            if income_level == 'Low':
                base_response_prob -= 0.1   # Socioeconomic bias
            if insurance_type == 'Uninsured':
                base_response_prob -= 0.2   # Access to care bias
            if region == 'Rural':
                base_response_prob -= 0.1   # Geographic bias
                
            # Add some genetic influence
            genetic_score = sum(genetic_markers.values()) / len(genetic_markers)
            base_response_prob += (genetic_score - 0.3) * 0.3
            
            # Ensure probability bounds
            base_response_prob = max(0.1, min(0.9, base_response_prob))
            
            treatment_response = np.random.binomial(1, base_response_prob)
            
            # Compile patient record
            patient_data = {
                'patient_id': f'P_{i:05d}',
                'age': round(age, 1),
                'gender': gender,
                'ethnicity': ethnicity,
                'income_level': income_level,
                'insurance_type': insurance_type,
                'region': region,
                'cancer_type': cancer_type,
                'stage': stage,
                'treatment_response': treatment_response,
                **genetic_markers
            }
            
            data.append(patient_data)
        
        return pd.DataFrame(data)
    
    def analyze_demographic_biases(self, df):
        """
        Analyze biases in the dataset across different demographic groups.
        
        Args:
            df (pd.DataFrame): Patient data
            
        Returns:
            dict: Bias analysis results
        """
        bias_analysis = {}
        
        # Response rates by ethnicity
        ethnicity_bias = df.groupby('ethnicity')['treatment_response'].agg(['mean', 'count']).round(3)
        bias_analysis['ethnicity_bias'] = ethnicity_bias
        
        # Response rates by income level
        income_bias = df.groupby('income_level')['treatment_response'].agg(['mean', 'count']).round(3)
        bias_analysis['income_bias'] = income_bias
        
        # Response rates by insurance type
        insurance_bias = df.groupby('insurance_type')['treatment_response'].agg(['mean', 'count']).round(3)
        bias_analysis['insurance_bias'] = insurance_bias
        
        # Response rates by region
        region_bias = df.groupby('region')['treatment_response'].agg(['mean', 'count']).round(3)
        bias_analysis['region_bias'] = region_bias
        
        # Gender bias
        gender_bias = df.groupby('gender')['treatment_response'].agg(['mean', 'count']).round(3)
        bias_analysis['gender_bias'] = gender_bias
        
        # Age bias (by age groups)
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 80, 100], 
                               labels=['<40', '40-60', '60-80', '>80'])
        age_bias = df.groupby('age_group')['treatment_response'].agg(['mean', 'count']).round(3)
        bias_analysis['age_bias'] = age_bias
        
        # Calculate bias severity scores
        bias_analysis['bias_scores'] = self.calculate_bias_scores(bias_analysis)
        
        return bias_analysis
    
    def calculate_bias_scores(self, bias_analysis):
        """
        Calculate bias severity scores for different demographic factors.
        
        Args:
            bias_analysis (dict): Bias analysis results
            
        Returns:
            dict: Bias severity scores
        """
        bias_scores = {}
        
        for category, data in bias_analysis.items():
            if category != 'bias_scores':
                if isinstance(data, pd.DataFrame) and 'mean' in data.columns:
                    # Calculate coefficient of variation as bias measure
                    response_rates = data['mean'].values
                    if len(response_rates) > 1:
                        bias_score = np.std(response_rates) / np.mean(response_rates)
                        bias_scores[category] = round(bias_score, 3)
        
        return bias_scores
    
    def train_biased_model(self, df):
        """
        Train a model on the biased dataset to demonstrate bias propagation.
        
        Args:
            df (pd.DataFrame): Patient data
            
        Returns:
            tuple: (model, performance metrics)
        """
        # Prepare features
        feature_columns = ['age', 'stage'] + [f'marker_{i}' for i in range(20)]
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_ethnicity = LabelEncoder()
        le_cancer = LabelEncoder()
        le_income = LabelEncoder()
        le_insurance = LabelEncoder()
        le_region = LabelEncoder()
        
        df_encoded = df.copy()
        df_encoded['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df_encoded['ethnicity_encoded'] = le_ethnicity.fit_transform(df['ethnicity'])
        df_encoded['cancer_type_encoded'] = le_cancer.fit_transform(df['cancer_type'])
        df_encoded['income_level_encoded'] = le_income.fit_transform(df['income_level'])
        df_encoded['insurance_type_encoded'] = le_insurance.fit_transform(df['insurance_type'])
        df_encoded['region_encoded'] = le_region.fit_transform(df['region'])
        
        # Features including demographic information
        all_features = feature_columns + [
            'gender_encoded', 'ethnicity_encoded', 'cancer_type_encoded',
            'income_level_encoded', 'insurance_type_encoded', 'region_encoded'
        ]
        
        X = df_encoded[all_features]
        y = df_encoded['treatment_response']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Performance metrics
        performance = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': dict(zip(all_features, self.model.feature_importances_))
        }
        
        return performance, X_test, y_test, y_pred, df_encoded.iloc[y_test.index]
    
    def fairness_strategies(self):
        """
        Define comprehensive fairness strategies for personalized medicine AI.
        
        Returns:
            dict: Fairness strategies and implementation guidelines
        """
        strategies = {
            "data_collection_strategies": {
                "diverse_recruitment": {
                    "description": "Actively recruit participants from underrepresented groups",
                    "implementation": [
                        "Partner with community health centers in diverse areas",
                        "Translate consent forms and materials into multiple languages",
                        "Provide transportation and childcare support for participation",
                        "Collaborate with minority-serving medical institutions"
                    ],
                    "target_metrics": "Achieve representation within 5% of national demographics"
                },
                "geographic_diversity": {
                    "description": "Ensure representation from rural, suburban, and urban areas",
                    "implementation": [
                        "Establish data collection sites in rural areas",
                        "Use telemedicine for remote participation",
                        "Account for regional health variations",
                        "Include environmental and lifestyle factors"
                    ]
                },
                "socioeconomic_inclusion": {
                    "description": "Include patients across all income levels and insurance types",
                    "implementation": [
                        "Provide financial incentives for participation",
                        "Cover costs associated with participation",
                        "Work with safety-net hospitals",
                        "Include uninsured and underinsured populations"
                    ]
                }
            },
            
            "algorithmic_fairness_techniques": {
                "pre_processing": {
                    "data_augmentation": "Generate synthetic data for underrepresented groups",
                    "resampling": "Balance dataset through stratified sampling",
                    "feature_selection": "Remove or de-weight biased features"
                },
                "in_processing": {
                    "fairness_constraints": "Add fairness constraints during model training",
                    "adversarial_debiasing": "Train models to be robust against demographic information",
                    "multi_task_learning": "Train separate models for different demographic groups"
                },
                "post_processing": {
                    "threshold_optimization": "Adjust decision thresholds for different groups",
                    "calibration": "Ensure prediction confidence is consistent across groups",
                    "output_correction": "Apply fairness-aware post-processing to predictions"
                }
            },
            
            "validation_and_monitoring": {
                "bias_testing": {
                    "description": "Continuous monitoring for biased outcomes",
                    "metrics": [
                        "Demographic parity: Equal positive prediction rates",
                        "Equalized odds: Equal true positive and false positive rates",
                        "Calibration: Equal precision across groups",
                        "Individual fairness: Similar individuals receive similar predictions"
                    ]
                },
                "external_validation": {
                    "description": "Test models on external, diverse datasets",
                    "implementation": [
                        "Collaborate with international research institutions",
                        "Validate on different healthcare systems",
                        "Test across different time periods",
                        "Include real-world evidence studies"
                    ]
                }
            },
            
            "governance_and_oversight": {
                "ethics_committees": {
                    "description": "Establish diverse ethics review boards",
                    "composition": [
                        "Healthcare professionals from diverse backgrounds",
                        "Community representatives",
                        "Ethicists and social scientists",
                        "Patient advocates",
                        "AI and data science experts"
                    ]
                },
                "transparency_measures": {
                    "description": "Ensure transparency in AI decision-making",
                    "implementation": [
                        "Provide explainable AI interfaces for clinicians",
                        "Publish bias testing results publicly",
                        "Document model limitations and contraindications",
                        "Enable patient access to AI decision factors"
                    ]
                },
                "regulatory_compliance": {
                    "description": "Ensure compliance with fairness regulations",
                    "standards": [
                        "FDA guidance on AI/ML in medical devices",
                        "EU AI Act requirements for high-risk AI systems",
                        "WHO ethics guidelines for AI in health",
                        "Professional medical association standards"
                    ]
                }
            },
            
            "clinical_implementation": {
                "provider_training": {
                    "description": "Train healthcare providers on AI bias awareness",
                    "topics": [
                        "Understanding AI limitations and biases",
                        "Interpreting AI recommendations appropriately",
                        "Recognizing when to override AI suggestions",
                        "Communicating uncertainty to patients"
                    ]
                },
                "patient_engagement": {
                    "description": "Engage patients in AI-assisted treatment decisions",
                    "approaches": [
                        "Explain AI recommendations in understandable terms",
                        "Discuss limitations and potential biases",
                        "Ensure informed consent for AI-assisted care",
                        "Provide options for human-only decision making"
                    ]
                }
            }
        }
        
        return strategies
    
    def visualize_bias_analysis(self, bias_analysis, df):
        """
        Create visualizations showing bias patterns in the dataset.
        
        Args:
            bias_analysis (dict): Bias analysis results
            df (pd.DataFrame): Patient data
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Treatment response by ethnicity
        ethnicity_data = bias_analysis['ethnicity_bias']
        axes[0, 0].bar(ethnicity_data.index, ethnicity_data['mean'], color='skyblue')
        axes[0, 0].set_title('Treatment Response Rate by Ethnicity')
        axes[0, 0].set_ylabel('Response Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Treatment response by income level
        income_data = bias_analysis['income_bias']
        axes[0, 1].bar(income_data.index, income_data['mean'], color='lightgreen')
        axes[0, 1].set_title('Treatment Response Rate by Income Level')
        axes[0, 1].set_ylabel('Response Rate')
        
        # 3. Treatment response by insurance type
        insurance_data = bias_analysis['insurance_bias']
        axes[0, 2].bar(insurance_data.index, insurance_data['mean'], color='lightcoral')
        axes[0, 2].set_title('Treatment Response Rate by Insurance Type')
        axes[0, 2].set_ylabel('Response Rate')
        
        # 4. Sample size by ethnicity
        axes[1, 0].bar(ethnicity_data.index, ethnicity_data['count'], color='orange')
        axes[1, 0].set_title('Sample Size by Ethnicity')
        axes[1, 0].set_ylabel('Number of Patients')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Bias severity scores
        bias_scores = bias_analysis['bias_scores']
        categories = list(bias_scores.keys())
        scores = list(bias_scores.values())
        axes[1, 1].bar(categories, scores, color='purple', alpha=0.7)
        axes[1, 1].set_title('Bias Severity Scores')
        axes[1, 1].set_ylabel('Coefficient of Variation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Age distribution by ethnicity
        for ethnicity in df['ethnicity'].unique():
            subset = df[df['ethnicity'] == ethnicity]
            axes[1, 2].hist(subset['age'], alpha=0.5, label=ethnicity, bins=20)
        axes[1, 2].set_title('Age Distribution by Ethnicity')
        axes[1, 2].set_xlabel('Age')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('personalized_medicine_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to demonstrate ethics analysis in personalized medicine.
    """
    print("⚕️ Ethics in Personalized Medicine AI Analysis")
    print("=" * 55)
    
    # Initialize analyzer
    analyzer = PersonalizedMedicineEthicsAnalyzer()
    
    # Generate synthetic genomic data
    print("🧬 Generating synthetic genomic dataset...")
    df = analyzer.generate_synthetic_genomic_data(n_samples=5000)
    print(f"Generated data for {len(df)} patients")
    print(f"Features: {list(df.columns)}")
    
    # Analyze demographic biases
    print("\n📊 Analyzing demographic biases...")
    bias_analysis = analyzer.analyze_demographic_biases(df)
    
    print("\nBias Analysis Results:")
    print("-" * 25)
    
    print("\n1. Treatment Response by Ethnicity:")
    print(bias_analysis['ethnicity_bias'])
    
    print("\n2. Treatment Response by Income Level:")
    print(bias_analysis['income_bias'])
    
    print("\n3. Treatment Response by Insurance Type:")
    print(bias_analysis['insurance_bias'])
    
    print("\n4. Bias Severity Scores:")
    for category, score in bias_analysis['bias_scores'].items():
        print(f"   {category}: {score}")
    
    # Train biased model
    print("\n🤖 Training AI model on biased dataset...")
    performance, X_test, y_test, y_pred, test_demographics = analyzer.train_biased_model(df)
    
    print(f"\nModel Performance:")
    print(f"  Overall Accuracy: {performance['classification_report']['accuracy']:.3f}")
    print(f"  Precision: {performance['classification_report']['macro avg']['precision']:.3f}")
    print(f"  Recall: {performance['classification_report']['macro avg']['recall']:.3f}")
    
    # Fairness strategies
    print("\n🛡️ Generating fairness strategies...")
    strategies = analyzer.fairness_strategies()
    
    print("\nKey Fairness Strategies:")
    print("-" * 25)
    print("1. Data Collection:")
    for strategy, details in strategies['data_collection_strategies'].items():
        print(f"   • {strategy}: {details['description']}")
    
    print("\n2. Algorithmic Fairness:")
    for approach, techniques in strategies['algorithmic_fairness_techniques'].items():
        print(f"   • {approach.replace('_', ' ').title()}:")
        for technique, description in techniques.items():
            print(f"     - {technique.replace('_', ' ').title()}: {description}")
    
    print("\n3. Governance & Oversight:")
    for measure, details in strategies['governance_and_oversight'].items():
        print(f"   • {measure.replace('_', ' ').title()}: {details['description']}")
    
    # Visualize bias analysis
    print("\n📈 Creating bias analysis visualizations...")
    analyzer.visualize_bias_analysis(bias_analysis, df)
    
    # Save detailed analysis
    with open('bias_analysis_report.txt', 'w') as f:
        f.write("PERSONALIZED MEDICINE AI BIAS ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY:\n")
        f.write("This analysis reveals significant biases in personalized medicine AI systems ")
        f.write("that could lead to disparate treatment outcomes across demographic groups.\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("1. Ethnic Minorities: Lower treatment response rates observed for African American, ")
        f.write("Hispanic, and Native American patients.\n")
        f.write("2. Socioeconomic Bias: Patients with low income and no insurance show worse outcomes.\n")
        f.write("3. Geographic Disparities: Rural patients have lower treatment success rates.\n")
        f.write("4. Sample Bias: Caucasian patients are overrepresented in the dataset.\n\n")
        
        f.write("RISKS:\n")
        f.write("- AI models trained on biased data will perpetuate and amplify existing disparities\n")
        f.write("- Underrepresented groups may receive suboptimal treatment recommendations\n")
        f.write("- Healthcare inequities could worsen with widespread AI adoption\n")
        f.write("- Trust in AI-assisted healthcare may be undermined\n\n")
        
        f.write("RECOMMENDED MITIGATION STRATEGIES:\n")
        f.write("1. Diversify training datasets through targeted recruitment\n")
        f.write("2. Implement algorithmic fairness techniques during model development\n")
        f.write("3. Establish continuous bias monitoring and model updates\n")
        f.write("4. Create diverse ethics oversight committees\n")
        f.write("5. Ensure transparency and explainability in AI recommendations\n")
        f.write("6. Train healthcare providers on bias awareness and mitigation\n")
    
    print("\n✅ Ethics analysis completed!")
    print("📄 Detailed report saved as 'bias_analysis_report.txt'")
    print("📊 Visualizations saved as 'personalized_medicine_bias_analysis.png'")

if __name__ == "__main__":
    main()
