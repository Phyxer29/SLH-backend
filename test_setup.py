import os
import sys
import requests
from dotenv import load_dotenv
import json

def test_setup():
    """Comprehensive setup validation"""
    
    print("\n" + "="*60)
    print("🔍 SHL SYSTEM SETUP VALIDATION")
    print("="*60 + "\n")
    
    load_dotenv()
    
    passed = 0
    failed = 0
    
    # Test 1: Environment Variables
    print("1️⃣ Checking Environment Variables...")
    
    keys_to_check = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")
    }
    
    for key, value in keys_to_check.items():
        if value:
            print(f"   ✅ {key}: {'*' * 10}{value[-4:]}")
            passed += 1
        else:
            print(f"   ❌ {key}: NOT SET")
            failed += 1
    
    # Test 2: Scraped Data
    print("\n2️⃣ Checking Scraped Data...")
    if os.path.exists("data/assessments.json"):
        with open("data/assessments.json") as f:
            data = json.load(f)
        
        count = len(data)
        
        if count >= 377:
            print(f"   ✅ Assessments: {count} (minimum 377 met)")
            passed += 1
        else:
            print(f"   ❌ Assessments: {count} (minimum 377 NOT met)")
            failed += 1
        
        # Check types
        types = {}
        for item in data:
            for t in item['test_type']:
                types[t] = types.get(t, 0) + 1
        
        print(f"\n   📊 Type Distribution:")
        for test_type, cnt in sorted(types.items(), key=lambda x: -x[1]):
            print(f"      - {test_type}: {cnt}")
    else:
        print("   ❌ assessments.json NOT FOUND")
        print("      Run: uvicorn app.main:app to trigger scraping")
        failed += 1
    
    # Test 3: Train/Test Data
    print("\n3️⃣ Checking Train/Test Data...")
    
    for filename in ["train.csv", "test.csv"]:
        path = f"data/{filename}"
        if os.path.exists(path):
            print(f"   ✅ {filename}: Found")
            passed += 1
        else:
            print(f"   ⚠ {filename}: Missing (download from assignment)")
            print(f"      Place in backend/data/{filename}")
    
    # Test 4: ChromaDB
    print("\n4️⃣ Checking ChromaDB...")
    
    if os.path.exists("chroma_db"):
        print(f"   ✅ Vector database initialized")
        passed += 1
    else:
        print(f"   ⚠ Not initialized yet (will be created on first run)")
    
    # Test 5: API Health Check
    print("\n5️⃣ Testing API Endpoints...")
    
    try:
        resp = requests.get("http://localhost:8000/health", timeout=5)
        
        if resp.status_code == 200:
            print(f"   ✅ Health endpoint: {resp.json()}")
            passed += 1
        else:
            print(f"   ❌ Health endpoint returned {resp.status_code}")
            failed += 1
    
    except requests.exceptions.ConnectionError:
        print("   ⚠ API not running")
        print("      Start with: uvicorn app.main:app --reload")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        failed += 1
    
    # Test 6: Recommendation Endpoint
    try:
        resp = requests.post(
            "http://localhost:8000/recommend",
            json={"query": "Java developer with team leadership skills"},
            timeout=15
        )
        
        if resp.status_code == 200:
            data = resp.json()
            count = len(data.get('recommended_assessments', []))
            
            if count >= 5:
                print(f"   ✅ Recommendation endpoint: {count} results")
                
                # Check balancing
                types = []
                for assessment in data['recommended_assessments']:
                    types.extend(assessment.get('test_type', []))
                
                has_knowledge = any('Knowledge' in t or 'Cognitive' in t for t in types)
                has_personality = any('Personality' in t for t in types)
                
                if has_knowledge and has_personality:
                    print(f"   ✅ Result balancing: Contains both technical and behavioral")
                else:
                    print(f"   ⚠ Result balancing: Check if mixed types present")
                
                passed += 1
            else:
                print(f"   ❌ Only {count} results (minimum 5 required)")
                failed += 1
        else:
            print(f"   ❌ Endpoint returned {resp.status_code}")
            failed += 1
    
    except requests.exceptions.ConnectionError:
        print("   ⚠ API not running")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        failed += 1
    
    # Final Summary
    print("\n" + "="*60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 All checks passed! System ready for submission.")
        return 0
    else:
        print("⚠ Some issues found. Fix them before submission.")
        return 1

if __name__ == "__main__":
    sys.exit(test_setup())
