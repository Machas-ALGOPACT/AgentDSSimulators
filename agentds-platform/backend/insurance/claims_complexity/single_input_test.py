"""
UI Input Testing Script

Demonstrates how to test the claims complexity model with single inputs.
This shows what a UI would need to collect from the user.

Usage:
    python single_input_test.py
"""

import os
import sys
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# SAMPLE INPUT DATA FOR UI
# ============================================================================
# These are the inputs a user would provide through a UI form

SAMPLE_CLAIMS = [
    {
        "name": "Minor Parking Lot Damage",
        "data": {
            # Required from UI form
            "ClaimID": "CLM-TEST-001",
            "PolicyID": "POL-TEST-001",
            "ClaimDate": "2026-01-08 14:30",
            "ClaimType": "Collision/Comprehensive",
            "ReportedDamage": 500.50,
            "NumParties": 1,
            "Description": "Minor parking lot damage to rear bumper. Single vehicle incident. No injuries. Damage estimate $500. Clean-up required.",
            
            # Optional - will be imputed if missing
            "HolderAge": 35.0,
            "VehicleType": "Sedan",
            "AnnualMileage": 8000.0,
            "LocationUrban": 1,  # 1=Urban, 0=Rural
            "CreditScore": 0.75,  # 0-1 scale
            "PolicyStart": "2023-01-08",
            "PolicyEnd": "2024-01-08",
        }
    },
    {
        "name": "Complex Multi-Vehicle Accident",
        "data": {
            # Required from UI form
            "ClaimID": "CLM-TEST-002",
            "PolicyID": "POL-TEST-002",
            "ClaimDate": "2026-01-07 09:15",
            "ClaimType": "Liability/Bodily Injury",
            "ReportedDamage": 15000.00,
            "NumParties": 4,
            "Description": "Multi-vehicle accident on highway involving 4 vehicles. Multiple injuries reported. Complex liability determination required. Extensive property damage. Police report filed. Multiple witnesses present. Investigation ongoing. Potential litigation.",
            
            # Optional - will be imputed if missing
            "HolderAge": 28.0,
            "VehicleType": "SUV",
            "AnnualMileage": 25000.0,
            "LocationUrban": 0,  # 1=Urban, 0=Rural
            "CreditScore": 0.45,  # 0-1 scale
            "PolicyStart": "2022-06-15",
            "PolicyEnd": "2025-06-15",
        }
    },
    {
        "name": "Moderate Theft Claim",
        "data": {
            # Required from UI form
            "ClaimID": "CLM-TEST-003",
            "PolicyID": "POL-TEST-003",
            "ClaimDate": "2026-01-06 22:45",
            "ClaimType": "Theft/Comprehensive",
            "ReportedDamage": 8500.75,
            "NumParties": 1,
            "Description": "Vehicle theft reported at parking garage. Vehicle recovered partially damaged. Some items missing from inside. Police report filed. Investigation in progress. Moderate complexity claim with recovery complications.",
            
            # Optional - will be imputed if missing (missing on purpose to test imputation)
            "HolderAge": None,
            "VehicleType": None,
            "AnnualMileage": None,
            "LocationUrban": None,
            "CreditScore": None,
            "PolicyStart": None,
            "PolicyEnd": None,
        }
    },
    {
        "name": "Straightforward Collision",
        "data": {
            # Required from UI form
            "ClaimID": "CLM-TEST-004",
            "PolicyID": "POL-TEST-004",
            "ClaimDate": "2026-01-05 11:20",
            "ClaimType": "Collision/Comprehensive",
            "ReportedDamage": 3200.00,
            "NumParties": 2,
            "Description": "Two vehicle rear-end collision at traffic light. Minor injuries. Damage clear and straightforward. One at-fault party. Simple liability determination. Quick resolution expected.",
            
            # Optional - will be imputed if missing
            "HolderAge": 45.0,
            "VehicleType": "Truck",
            "AnnualMileage": 12000.0,
            "LocationUrban": 1,  # 1=Urban, 0=Rural
            "CreditScore": 0.85,  # 0-1 scale
            "PolicyStart": "2021-03-20",
            "PolicyEnd": "2026-03-20",
        }
    },
]


# ============================================================================
# FIELD DOCUMENTATION FOR UI DEVELOPERS
# ============================================================================

FIELD_GUIDE = {
    "ClaimID": {
        "type": "text",
        "required": True,
        "example": "CLM-001234",
        "description": "Unique claim identifier"
    },
    "PolicyID": {
        "type": "text",
        "required": True,
        "example": "POL-001234",
        "description": "Unique policy identifier"
    },
    "ClaimDate": {
        "type": "datetime",
        "required": True,
        "format": "YYYY-MM-DD HH:MM",
        "example": "2026-01-08 14:30",
        "description": "Date and time claim was reported"
    },
    "ClaimType": {
        "type": "dropdown",
        "required": True,
        "options": [
            "Collision/Comprehensive",
            "Liability/Bodily Injury",
            "Theft/Comprehensive",
            "Fender-Bender",
            "Vandalism",
            "Other"
        ],
        "description": "Type of insurance claim"
    },
    "ReportedDamage": {
        "type": "number",
        "required": True,
        "min": 0,
        "example": 5000.50,
        "description": "Reported damage amount in dollars"
    },
    "NumParties": {
        "type": "integer",
        "required": True,
        "min": 1,
        "example": 2,
        "description": "Number of parties involved in the claim"
    },
    "Description": {
        "type": "textarea",
        "required": True,
        "min_length": 50,
        "example": "Brief description of what happened in the claim...",
        "description": "Detailed claim description (used for AI analysis)"
    },
    "HolderAge": {
        "type": "number",
        "required": False,
        "min": 16,
        "max": 120,
        "example": 35.0,
        "description": "Age of policy holder (imputed if missing)"
    },
    "VehicleType": {
        "type": "dropdown",
        "required": False,
        "options": [
            "Sedan",
            "SUV",
            "Truck",
            "Coupe",
            "Hatchback",
            "Minivan",
            "Motorcycle",
            "Other"
        ],
        "description": "Type of vehicle (imputed if missing)"
    },
    "AnnualMileage": {
        "type": "number",
        "required": False,
        "min": 0,
        "example": 10000.0,
        "description": "Annual mileage driven (imputed if missing)"
    },
    "LocationUrban": {
        "type": "radio",
        "required": False,
        "options": {"1": "Urban", "0": "Rural"},
        "example": 1,
        "description": "Location type (1=Urban, 0=Rural, imputed if missing)"
    },
    "CreditScore": {
        "type": "number",
        "required": False,
        "min": 0,
        "max": 1,
        "example": 0.75,
        "description": "Credit score on 0-1 scale (imputed if missing)"
    },
    "PolicyStart": {
        "type": "date",
        "required": False,
        "format": "YYYY-MM-DD",
        "example": "2023-01-08",
        "description": "Policy start date (imputed if missing)"
    },
    "PolicyEnd": {
        "type": "date",
        "required": False,
        "format": "YYYY-MM-DD",
        "example": "2024-01-08",
        "description": "Policy end date (imputed if missing)"
    },
}


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_single_claim(claim_data):
    """
    Test single claim prediction
    
    Args:
        claim_data (dict): Single claim input data
        
    Returns:
        dict: Prediction result with confidence scores
    """
    try:
        # Load trained artifacts
        model_path = os.path.join(current_dir, 'models', 'ensemble_model.joblib')
        pipeline_path = os.path.join(current_dir, 'models', 'preprocessing_pipeline.joblib')
        encoder_path = os.path.join(current_dir, 'models', 'label_encoder.joblib')
        
        if not all(os.path.exists(p) for p in [model_path, pipeline_path, encoder_path]):
            logger.error("Model artifacts not found. Run main.py first to train models.")
            return None
        
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        label_encoder = joblib.load(encoder_path)
        
        # Prepare input as DataFrame
        df = pd.DataFrame([claim_data])
        
        # Transform features using pipeline
        X = pipeline.transform(df)
        
        # Make prediction
        prediction_numeric = model.predict(X)[0]
        prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]
        
        # Get confidence scores
        probabilities = model.predict_proba(X)[0]
        
        result = {
            "prediction": prediction_label,
            "confidence": float(probabilities[prediction_numeric]),
            "probabilities": {
                label: float(prob) 
                for label, prob in zip(label_encoder.classes_, probabilities)
            },
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CLAIMS COMPLEXITY MODEL - UI INPUT TEST")
    print("="*80)
    
    print("\n\nFIELD GUIDE FOR UI DEVELOPERS:")
    print("-" * 80)
    print("\nRequired Fields (must be provided by user):")
    required_fields = [f for f, info in FIELD_GUIDE.items() if info.get("required", False)]
    for field in required_fields:
        info = FIELD_GUIDE[field]
        print(f"\n  {field}")
        print(f"    Type: {info['type']}")
        if 'format' in info:
            print(f"    Format: {info['format']}")
        print(f"    Description: {info['description']}")
        print(f"    Example: {info.get('example', 'N/A')}")
        if 'options' in info and isinstance(info['options'], list):
            print(f"    Options: {', '.join(info['options'][:3])}...")
    
    print("\n\nOptional Fields (will be imputed if missing):")
    optional_fields = [f for f, info in FIELD_GUIDE.items() if not info.get("required", False)]
    for field in optional_fields:
        info = FIELD_GUIDE[field]
        print(f"  {field} ({info['type']}) - {info['description']}")
    
    # Run predictions on sample data
    print("\n\n" + "="*80)
    print("RUNNING PREDICTIONS ON SAMPLE DATA")
    print("="*80)
    
    for i, claim_sample in enumerate(SAMPLE_CLAIMS, 1):
        print(f"\n\nTest {i}: {claim_sample['name']}")
        print("-" * 80)
        
        # Show input
        print("\nInput Data:")
        for key, value in claim_sample['data'].items():
            if key == 'Description':
                desc = value[:60] + "..." if len(str(value)) > 60 else value
                print(f"  {key}: {desc}")
            else:
                print(f"  {key}: {value}")
        
        # Get prediction
        result = test_single_claim(claim_sample['data'])
        
        # Show results
        if result and result.get('status') == 'success':
            print("\nPrediction Result:")
            print(f"  Predicted Complexity: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"\n  Probability Breakdown:")
            for label, prob in result['probabilities'].items():
                bar_length = int(prob * 30)
                bar = "#" * bar_length + "-" * (30 - bar_length)
                print(f"    {label:12} {bar} {prob:.1%}")
        else:
            print(f"\nError: {result.get('message', 'Unknown error')}")
    
    # UI Input form template
    print("\n\n" + "="*80)
    print("UI FORM TEMPLATE (What Your UI Should Collect)")
    print("="*80)
    
    print("\nCLAIMS COMPLEXITY PREDICTION - USER FORM\n")
    print("Required Information:")
    print("  Claim ID              [____________]")
    print("  Policy ID             [____________]")
    print("  Claim Date            [__/__/____ __:__]")
    print("  Claim Type            [v Dropdown Selection]")
    print("  Reported Damage ($)   [____________]")
    print("  Number of Parties     [____________]")
    print("  Claim Description:    (50+ characters)")
    print("    (Detailed description of what happened)")
    print("\nOptional Information (leave blank if unknown):")
    print("  Policyholder Age      [____________]")
    print("  Vehicle Type          [v Dropdown]")
    print("  Annual Mileage        [____________]")
    print("  Location              ( Urban  ( Rural")
    print("  Credit Score (0-1)    [____________]")
    print("  Policy Start Date     [__/__/____]")
    print("  Policy End Date       [__/__/____]")
    print("\n  [PREDICT COMPLEXITY]")
    print("\nOutput:")
    print("  Predicted Complexity: SIMPLE / MODERATE / COMPLEX")
    print("  Confidence: 86%")
    print("  Breakdown:")
    print("    Simple:    ####################---- 86%")
    print("    Moderate:  ######------------------ 10%")
    print("    Complex:   ----------------------- 4%")
    
    print("\n" + "="*80)
    print("KEY NOTES FOR UI DEVELOPERS")
    print("="*80)
    print("\n1. REQUIRED FIELDS (must validate):")
    print("   - ClaimID, PolicyID, ClaimDate, ClaimType")
    print("   - ReportedDamage, NumParties, Description\n")
    print("2. DESCRIPTION FIELD:")
    print("   - Minimum 50 characters required")
    print("   - This is the most important field for prediction")
    print("   - Quality descriptions improve prediction accuracy\n")
    print("3. OPTIONAL FIELDS:")
    print("   - If missing, system auto-imputes with median/mode values")
    print("   - Can be left blank by user")
    print("   - System will fill with reasonable defaults\n")
    print("4. PREDICTION OUTPUT:")
    print("   - Confidence indicates how certain the model is")
    print("   - Probability breakdown helps understand the decision")
    print("   - Three classes: Simple (80%), Moderate (15%), Complex (5%)\n")
    print("5. BACKEND API ENDPOINT:")
    print("   - POST /claims/predict")
    print("   - Send JSON with all fields above")
    print("   - Returns: {complexity, confidence, probabilities}\n")
    print("6. DATA VALIDATION:")
    print("   - ClaimDate must be valid datetime")
    print("   - ReportedDamage must be positive number")
    print("   - NumParties must be positive integer")
    print("   - CreditScore between 0-1 if provided")
    print("   - HolderAge between 16-120 if provided")


if __name__ == "__main__":
    main()
