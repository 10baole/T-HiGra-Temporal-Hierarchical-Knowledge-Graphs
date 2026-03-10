import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import dateparser
from higra_agent.retriever.higra_retriever.higra_schema import TemporalInterval


class TemporalNormalizer:
    def __init__(self):
        # Common temporal patterns
        self.year_pattern = r'\b(1[0-9]{3}|2[0-9]{3})\b'
        self.date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        self.range_pattern = r'\b(\d{4})\s*(?:to|-)\s*(\d{4})\b'
        
    def extract_temporal_expressions(self, text: str) -> List[Dict[str, Any]]:
        expressions = []
        
        # Extract year ranges
        for match in re.finditer(self.range_pattern, text, re.IGNORECASE):
            start_year = match.group(1)
            end_year   = match.group(2)
            expressions.append({
                'text': match.group(0),
                'type': 'range',
                'start': f"{start_year}-01-01",
                'end': f"{end_year}-12-31",
                'span': match.span()
            })
        
        # Extract individual years
        for match in re.finditer(self.year_pattern, text):
            year = match.group(0)
            # Skip if already part of a range
            if not any(expr['span'][0] <= match.start() <= expr['span'][1] for expr in expressions):
                expressions.append({
                    'text': year,
                    'type': 'year',
                    'start': f"{year}-01-01",
                    'end': f"{year}-12-31",
                    'span': match.span()
                })
        
        # Extract dates
        for match in re.finditer(self.date_pattern, text):
            date_str = match.group(0)
            parsed = dateparser.parse(date_str)
            if parsed:
                expressions.append({
                    'text': date_str,
                    'type': 'date',
                    'start': parsed.date().isoformat(),
                    'end': parsed.date().isoformat(),
                    'span': match.span()
                })
        
        return expressions
    
    def normalize_to_interval(self, temporal_expr: Dict[str, Any], default_end: Optional[str] = None) -> TemporalInterval:

        if temporal_expr['type'] == 'range':
            return TemporalInterval(
                start=temporal_expr['start'],
                end=temporal_expr['end'],
                confidence=0.9
            )
        elif temporal_expr['type'] in ['year', 'date']:
            return TemporalInterval(
                start=temporal_expr['start'],
                end=temporal_expr.get('end', default_end),
                confidence=0.8
            )
        
        return TemporalInterval(confidence=0.5)
    
    def expand_to_interval(self, timestamp) -> TemporalInterval:
        if isinstance(timestamp, TemporalInterval):
            return timestamp
        
        if not timestamp:
            return None
            
        normalized = timestamp.replace('/', '-')
        parts = normalized.split('-')
        
        if len(parts) == 1:  # Year only: "2007"
            year = parts[0]
            return TemporalInterval(
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                confidence=1.0
            )
        elif len(parts) == 2:  # Year-Month: "2007-01"
            year, month = parts
            # Get last day of month
            if month == '02':
                last_day = '28' if int(year) % 4 != 0 else '29'
            elif month in ['04', '06', '09', '11']:
                last_day = '30'
            else:
                last_day = '31'
            return TemporalInterval(
                start=f"{year}-{month}-01",
                end=f"{year}-{month}-{last_day}",
                confidence=1.0
            )
        else:  # Full date
            return TemporalInterval(
                start=timestamp,
                end=timestamp,
                confidence=1.0
            )

class TemporalValidator: 
    def __init__(self):
        pass
    
    def _parse_date(self, date_str: str) -> datetime:
        try:
            if len(date_str) == 4 and date_str.isdigit():  # year-only "2019"
                return datetime(int(date_str), 1, 1)
            if 'T' in date_str:  #  ISO format 
                return datetime.fromisoformat(date_str.split('T')[0])
            return datetime.fromisoformat(date_str)
        
        except:
            parsed = dateparser.parse(date_str)
            if parsed:
                return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
            raise ValueError(f"Cannot parse date: {date_str}")
    
    def validate_query_time(self, query_time: TemporalInterval, available_timestamps: List[str]) -> Dict[str, Any]:
        if not available_timestamps:
            return {
                "is_valid": False,
                "should_refuse": True,
                "reason": "no_temporal_data",
                "matching_timestamps": []
            }
        
        if not query_time or not query_time.start:
            return {
                "is_valid": True,
                "should_refuse": False,
                "reason": "no_temporal_constraint",
                "matching_timestamps": available_timestamps
            }
        
        query_start = self._parse_date(query_time.start)
        
        matching = []
        for ts in available_timestamps:
            try:
                ts_date = self._parse_date(ts)
                if ts_date <= query_start:
                    matching.append(ts)
            except Exception as e:
                print(f"Warning: Failed to parse timestamp {ts}: {e}")
        
        if not matching:
            return {
                "is_valid": False,
                "should_refuse": True,
                "reason": f"no_valid_context for timestamp {query_time.start}",
                "matching_timestamps": []
            }
        
        return {
            "is_valid": True,
            "should_refuse": False,
            "reason": "valid",
            "matching_timestamps": matching
        }