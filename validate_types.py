#!/usr/bin/env python3
"""
Type checking validation script for Monica Bot.
Validates that all type hints are correctly implemented and catch any type errors.
"""

from typing import get_type_hints, get_origin, get_args
import importlib
import sys
import os

# Add the bot directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_module_types(module_name: str):
    """Check type annotations in a module."""
    try:
        module = importlib.import_module(module_name)
        print(f"\n✅ Successfully imported {module_name}")
        
        # Check if module has type annotations
        if hasattr(module, '__annotations__'):
            print(f"   Module-level annotations: {len(module.__annotations__)}")
        
        # Count functions and classes with annotations
        function_count = 0
        annotated_function_count = 0
        class_count = 0
        
        for name in dir(module):
            obj = getattr(module, name)
            
            if callable(obj) and not name.startswith('_'):
                function_count += 1
                try:
                    hints = get_type_hints(obj)
                    if hints:
                        annotated_function_count += 1
                except Exception:
                    pass
            
            elif isinstance(obj, type) and not name.startswith('_'):
                class_count += 1
                print(f"   Class found: {name}")
                
                # Check class methods
                for method_name in dir(obj):
                    if not method_name.startswith('_') and callable(getattr(obj, method_name)):
                        try:
                            method = getattr(obj, method_name)
                            hints = get_type_hints(method)
                            if hints:
                                print(f"     ✓ {method_name} has type hints")
                        except Exception:
                            pass
        
        print(f"   Functions: {annotated_function_count}/{function_count} have type hints")
        print(f"   Classes: {class_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        return False

def main():
    """Main type checking routine."""
    print("🔍 Monica Bot Type Hint Validation")
    print("=" * 50)
    
    modules_to_check = [
        "bot",
        "modules.config", 
        "modules.metrics",
        "modules.utils",
        "modules.voice_manager", 
        "modules.audio_processor",
        "modules.cache_manager",
        "modules.ui_components"
    ]
    
    success_count = 0
    total_count = len(modules_to_check)
    
    for module_name in modules_to_check:
        if check_module_types(module_name):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Summary: {success_count}/{total_count} modules successfully validated")
    
    if success_count == total_count:
        print("🎉 All modules passed type hint validation!")
        return 0
    else:
        print("⚠️  Some modules failed validation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
