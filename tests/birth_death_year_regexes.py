# Regular expressions for matching birth and/or death years in name strings.
import re

def _compile_birth_death_pattern(patterns):
        """Compile regular expressions for birth-death year pattern matching"""
        patterns = []
        
        # Pattern 1: Birth year with approximate death year - "565 - approximately 665"
        patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 2: Approximate birth and death years
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 3: Approximate birth with standard death
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 4: Standard birth-death range
        patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 5: Death year only with approximate marker
        patterns.append(r'[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 6: Death year only (simple)
        patterns.append(r'[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 7: Birth year only with approximate marker
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
        
        # Pattern 8: Birth year only (simple)
        patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
        
        # Pattern 9: Explicit birth/death prefixes
        patterns.append(r'(?:b\.|born)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)|(?:d\.|died)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 10: Single approximate year (fallback)
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        return [re.compile(pattern) for pattern in patterns]