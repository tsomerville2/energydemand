"""
NAESB EDI Parser and Generator
Real implementation of NAESB X12 EDI transactions for energy nominations
Demonstrates EDI message creation, parsing, and validation
"""

import json
from datetime import datetime, timedelta
import re

print("=" * 60)
print("NAESB EDI TRANSACTION PROCESSOR")
print("=" * 60)

class NAESBEDIProcessor:
    """Process NAESB-compliant EDI messages"""
    
    def __init__(self):
        self.segment_terminators = {
            'segment': '~',
            'element': '*',
            'subelement': ':'
        }
        self.transaction_sets = {
            '866': 'NOMINATION_REQUEST',
            '824': 'CONFIRMATION_RESPONSE',
            '867': 'PRODUCT_TRANSFER',
            '997': 'FUNCTIONAL_ACKNOWLEDGMENT'
        }
        
    def create_isa_header(self, sender_id, receiver_id, control_num):
        """Create ISA (Interchange Control Header) segment"""
        isa = [
            'ISA',  # Segment ID
            '00',   # Authorization Information Qualifier
            '          ',  # Authorization Information (10 spaces)
            '00',   # Security Information Qualifier
            '          ',  # Security Information (10 spaces)
            'ZZ',   # Interchange ID Qualifier (ZZ = Mutually Defined)
            f"{sender_id:15}",  # Interchange Sender ID (15 chars)
            'ZZ',   # Interchange ID Qualifier
            f"{receiver_id:15}",  # Interchange Receiver ID (15 chars)
            datetime.now().strftime('%y%m%d'),  # Date (YYMMDD)
            datetime.now().strftime('%H%M'),    # Time (HHMM)
            'U',    # Interchange Control Standards ID
            '00401',  # Interchange Control Version Number (NAESB standard)
            f"{control_num:09d}",  # Interchange Control Number
            '0',    # Acknowledgment Requested
            'P',    # Usage Indicator (P = Production)
            ':'     # Component Element Separator
        ]
        return self.segment_terminators['element'].join(isa)
    
    def create_gs_header(self, functional_group, sender_code, receiver_code, control_num):
        """Create GS (Functional Group Header) segment"""
        gs = [
            'GS',
            functional_group,  # Functional Identifier Code
            sender_code,       # Application Sender's Code
            receiver_code,     # Application Receiver's Code
            datetime.now().strftime('%Y%m%d'),  # Date (YYYYMMDD)
            datetime.now().strftime('%H%M%S'),  # Time (HHMMSS)
            str(control_num),  # Group Control Number
            'X',               # Responsible Agency Code
            '004010'           # Version/Release/Industry ID Code
        ]
        return self.segment_terminators['element'].join(gs)
    
    def create_nomination_transaction(self, nomination_data):
        """Create 866 Transaction Set (Production Nomination)"""
        segments = []
        
        # ST - Transaction Set Header
        segments.append(f"ST*866*0001")
        
        # BSS - Beginning Segment for Scheduling
        segments.append(f"BSS*00*NOM*{nomination_data['transaction_id']}*"
                       f"{nomination_data['gas_day'].replace('-', '')}*"
                       f"{nomination_data['cycle']}")
        
        # DTM - Date/Time Reference (Nomination effective time)
        segments.append(f"DTM*007*{nomination_data['gas_day'].replace('-', '')}*0900")
        
        # N1 - Name (Shipper)
        segments.append(f"N1*SH*{nomination_data['shipper_name']}*"
                       f"1*{nomination_data['shipper_duns']}")
        
        # N1 - Name (Pipeline/TSP)
        segments.append(f"N1*PL*{nomination_data['pipeline_name']}*"
                       f"1*{nomination_data['pipeline_duns']}")
        
        # Loop for each location (receipt/delivery points)
        location_counter = 1
        for location in nomination_data['locations']:
            # PAL - Product Activity Location
            segments.append(f"PAL*{location_counter:03d}*"
                          f"{location['type']}*"
                          f"{location['code']}*"
                          f"{location['name']}")
            
            # QTY - Quantity
            segments.append(f"QTY*38*{location['quantity']}*DT")  # DT = Dekatherm
            
            # REF - Reference (Upstream/Downstream party)
            if location['type'] == 'R':  # Receipt
                segments.append(f"REF*UP*{location['upstream_party']}")
            else:  # Delivery
                segments.append(f"REF*DN*{location['downstream_party']}")
            
            location_counter += 1
        
        # CTT - Transaction Totals
        segments.append(f"CTT*{len(nomination_data['locations'])}")
        
        # SE - Transaction Set Trailer
        segments.append(f"SE*{len(segments) + 1}*0001")
        
        return segments
    
    def create_confirmation_response(self, confirmation_data):
        """Create 824 Transaction Set (Application Advice/Confirmation)"""
        segments = []
        
        # ST - Transaction Set Header
        segments.append(f"ST*824*0001")
        
        # BGN - Beginning Segment
        segments.append(f"BGN*11*{confirmation_data['confirmation_id']}*"
                       f"{datetime.now().strftime('%Y%m%d')}*"
                       f"{datetime.now().strftime('%H%M%S')}")
        
        # N1 - Name (Pipeline confirming)
        segments.append(f"N1*PL*{confirmation_data['pipeline_name']}")
        
        # OTI - Original Transaction Identification
        segments.append(f"OTI*IA*NA*{confirmation_data['original_transaction_id']}")
        
        # TED - Technical Error Description (or confirmation)
        if confirmation_data['status'] == 'CONFIRMED':
            segments.append(f"TED*000*NOMINATION CONFIRMED")
        else:
            segments.append(f"TED*001*{confirmation_data['reason']}")
        
        # QTY - Quantities
        segments.append(f"QTY*RQ*{confirmation_data['requested_quantity']}*DT")
        segments.append(f"QTY*SQ*{confirmation_data['scheduled_quantity']}*DT")
        
        # SE - Transaction Set Trailer
        segments.append(f"SE*{len(segments) + 1}*0001")
        
        return segments
    
    def build_complete_message(self, transaction_segments, sender_id, receiver_id):
        """Build complete EDI message with envelopes"""
        message = []
        
        # ISA Header
        message.append(self.create_isa_header(sender_id, receiver_id, 1))
        
        # GS Header
        message.append(self.create_gs_header('PN', sender_id, receiver_id, 1))
        
        # Transaction Set
        message.extend(transaction_segments)
        
        # GE Trailer
        message.append(f"GE*1*1")  # 1 transaction set, group control 1
        
        # IEA Trailer
        message.append(f"IEA*1*000000001")  # 1 functional group, control 1
        
        return self.segment_terminators['segment'].join(message) + self.segment_terminators['segment']
    
    def parse_edi_message(self, edi_string):
        """Parse EDI message into structured data"""
        segments = edi_string.strip().split(self.segment_terminators['segment'])
        parsed = {
            'header': {},
            'transactions': [],
            'valid': True,
            'errors': []
        }
        
        for segment in segments:
            if not segment:
                continue
                
            elements = segment.split(self.segment_terminators['element'])
            segment_id = elements[0]
            
            if segment_id == 'ISA':
                parsed['header']['sender_id'] = elements[6].strip()
                parsed['header']['receiver_id'] = elements[8].strip()
                parsed['header']['date'] = elements[9]
                parsed['header']['time'] = elements[10]
                parsed['header']['control_num'] = elements[13]
                
            elif segment_id == 'ST':
                transaction = {
                    'type': elements[1],
                    'control_num': elements[2],
                    'segments': []
                }
                parsed['transactions'].append(transaction)
                
            elif segment_id in ['BSS', 'DTM', 'N1', 'PAL', 'QTY', 'REF']:
                if parsed['transactions']:
                    parsed['transactions'][-1]['segments'].append({
                        'id': segment_id,
                        'data': elements[1:]
                    })
        
        # Validate structure
        if not parsed['header'].get('sender_id'):
            parsed['valid'] = False
            parsed['errors'].append('Missing ISA header')
            
        return parsed

# Create processor
processor = NAESBEDIProcessor()

# Example 1: Create a Nomination EDI Message
print("\n1. Creating NAESB EDI Nomination Message (866 Transaction Set):")
print("-" * 40)

nomination_data = {
    'transaction_id': 'NOM123456',
    'gas_day': '2025-08-28',
    'cycle': 'TIMELY',
    'shipper_name': 'EXAMPLE_GAS_MARKETING',
    'shipper_duns': '123456789',
    'pipeline_name': 'TRANSCANADA_PIPELINE',
    'pipeline_duns': '987654321',
    'locations': [
        {
            'type': 'R',  # Receipt
            'code': 'LOC001',
            'name': 'PERMIAN_HUB',
            'quantity': 50000,
            'upstream_party': 'PRODUCER_ABC'
        },
        {
            'type': 'D',  # Delivery
            'code': 'LOC002',
            'name': 'CHICAGO_CITYGATE',
            'quantity': 50000,
            'downstream_party': 'LDC_CHICAGO'
        }
    ]
}

# Create nomination transaction
nom_segments = processor.create_nomination_transaction(nomination_data)

# Build complete message
edi_nomination = processor.build_complete_message(
    nom_segments,
    'GASMARKETING001',
    'PIPELINE001'
)

print("Generated EDI Message Preview:")
print(edi_nomination[:200] + "...")
print(f"Total Message Length: {len(edi_nomination)} characters")
print(f"Number of segments: {edi_nomination.count('~')}")

# Save EDI message
with open('SCAN/energy_nominations/naesb_edi/nomination.edi', 'w') as f:
    f.write(edi_nomination)

# Example 2: Create Confirmation Response
print("\n2. Creating NAESB EDI Confirmation Response (824 Transaction Set):")
print("-" * 40)

confirmation_data = {
    'confirmation_id': 'CONF789012',
    'pipeline_name': 'TRANSCANADA_PIPELINE',
    'original_transaction_id': 'NOM123456',
    'status': 'CONFIRMED',
    'requested_quantity': 50000,
    'scheduled_quantity': 50000,
    'reason': 'FULLY_SCHEDULED'
}

# Create confirmation transaction
conf_segments = processor.create_confirmation_response(confirmation_data)

# Build complete message
edi_confirmation = processor.build_complete_message(
    conf_segments,
    'PIPELINE001',
    'GASMARKETING001'
)

print("Generated Confirmation EDI Message Preview:")
print(edi_confirmation[:200] + "...")
print(f"Total Message Length: {len(edi_confirmation)} characters")

# Save confirmation
with open('SCAN/energy_nominations/naesb_edi/confirmation.edi', 'w') as f:
    f.write(edi_confirmation)

# Example 3: Parse EDI Message
print("\n3. Parsing EDI Message:")
print("-" * 40)

parsed = processor.parse_edi_message(edi_nomination)

print(f"Message Valid: {'✅' if parsed['valid'] else '❌'}")
print(f"Sender ID: {parsed['header'].get('sender_id', 'N/A')}")
print(f"Receiver ID: {parsed['header'].get('receiver_id', 'N/A')}")
print(f"Date: {parsed['header'].get('date', 'N/A')}")
print(f"Number of Transactions: {len(parsed['transactions'])}")

if parsed['transactions']:
    trans = parsed['transactions'][0]
    print(f"Transaction Type: {trans['type']} ({processor.transaction_sets.get(trans['type'], 'UNKNOWN')})")
    print(f"Number of segments in transaction: {len(trans['segments'])}")

# Example 4: Create Human-Readable Report
print("\n4. Human-Readable Nomination Summary:")
print("-" * 40)

report = {
    'Transaction Details': {
        'ID': nomination_data['transaction_id'],
        'Gas Day': nomination_data['gas_day'],
        'Cycle': nomination_data['cycle'],
        'Shipper': nomination_data['shipper_name'],
        'Pipeline': nomination_data['pipeline_name']
    },
    'Locations': [],
    'Totals': {
        'Total Receipt': sum([loc['quantity'] for loc in nomination_data['locations'] if loc['type'] == 'R']),
        'Total Delivery': sum([loc['quantity'] for loc in nomination_data['locations'] if loc['type'] == 'D']),
        'Imbalance': 0
    }
}

for loc in nomination_data['locations']:
    report['Locations'].append({
        'Type': 'Receipt' if loc['type'] == 'R' else 'Delivery',
        'Location': loc['name'],
        'Quantity (DTH)': f"{loc['quantity']:,}",
        'Counterparty': loc.get('upstream_party', loc.get('downstream_party', 'N/A'))
    })

report['Totals']['Imbalance'] = report['Totals']['Total Receipt'] - report['Totals']['Total Delivery']

# Save report as JSON
with open('SCAN/energy_nominations/naesb_edi/nomination_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Print summary
for category, details in report.items():
    if category == 'Locations':
        print(f"\n{category}:")
        for loc in details:
            print(f"  {loc['Type']:8} | {loc['Location']:20} | {loc['Quantity (DTH)']:>10} DTH")
    elif category == 'Transaction Details':
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key:12} {value}")
    else:
        print(f"\n{category}:")
        for key, value in details.items():
            print(f"  {key:15} {value:,} DTH" if isinstance(value, int) else f"  {key:15} {value}")

print("\n" + "=" * 60)
print("NAESB EDI Processing Complete!")
print("Results saved to:")
print("  - SCAN/energy_nominations/naesb_edi/nomination.edi")
print("  - SCAN/energy_nominations/naesb_edi/confirmation.edi")
print("  - SCAN/energy_nominations/naesb_edi/nomination_report.json")
print("=" * 60)