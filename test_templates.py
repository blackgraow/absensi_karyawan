#!/usr/bin/env python
"""Quick template rendering test to verify no Jinja2 errors."""
import sys
sys.path.insert(0, '.')

from app import create_app
from jinja2 import TemplateAssertionError

try:
    app = create_app()
    
    # Test 1: Check if base.html can be parsed
    print("Testing base.html template parsing...")
    with app.app_context():
        template = app.jinja_env.get_template('base.html')
        print("✓ base.html parses successfully (no TemplateAssertionError)")
    
    # Test 2: Check if login.html extends base.html correctly
    print("\nTesting login.html template...")
    with app.app_context():
        template = app.jinja_env.get_template('login.html')
        print("✓ login.html parses successfully")
    
    # Test 3: Check if dashboard.html extends base.html correctly  
    print("\nTesting dashboard.html template...")
    with app.app_context():
        template = app.jinja_env.get_template('dashboard.html')
        print("✓ dashboard.html parses successfully")
    
    print("\n✅ All templates validated successfully!")
    
except TemplateAssertionError as e:
    print(f"❌ Template Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
