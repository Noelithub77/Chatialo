from bs4 import BeautifulSoup
import re

def clean_text(text):
    # Clean text by removing extra whitespace and newlines
    return ' '.join(text.strip().split())

def extract_email(text):
    # Extract email from text that uses " at " and " dot " format
    if 'at' in text and 'dot' in text:
        return text.replace(' at ', '@').replace(' dot ', '.')
    return text

def extract_faculty_details(html_file, output_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Find all faculty cards
    faculty_cards = soup.find_all('div', class_='card flex flex-column custom-card-depth border-radius')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for card in faculty_cards:
            # Extract faculty details
            name = card.find('h5', class_='white-text')
            position = card.find_all('div', class_='custom-card-sub-head bg-white border border-bottom border-dull')
            qualifications = card.find('b', class_='ng-binding')
            contact_info = card.find_all('div', class_='chip truncate transparent link-color ng-binding')
            
            if name and not name.text.isspace():
                f.write(f"\nFaculty Name: {clean_text(name.text)}\n")
                f.write("-" * 50 + "\n")

                # Write position/roles
                if position and len(position) > 1:
                    roles = clean_text(position[1].text)
                    if roles and not roles.isspace():
                        f.write(f"Position: {roles}\n")

                # Write additional roles/responsibilities
                if position and len(position) > 0:
                    additional_roles = clean_text(position[0].text)
                    if additional_roles and not additional_roles.isspace():
                        f.write(f"Additional Roles: {additional_roles}\n")

                # Write qualifications
                if qualifications and qualifications.text.strip():
                    f.write(f"Qualifications: {clean_text(qualifications.text)}\n")

                # Write contact information
                for contact in contact_info:
                    if 'phone' in str(contact).lower():
                        f.write(f"Phone: {clean_text(contact.text)}\n")
                    elif 'email' in str(contact).lower():
                        f.write(f"Email: {extract_email(clean_text(contact.text))}\n")
                    elif 'location' in str(contact).lower():
                        f.write(f"Room: {clean_text(contact.text)}\n")

                f.write("\n")

if __name__ == "__main__":
    input_html = "Faculty _ IIIT Kottayam.html"  # Your HTML file name
    output_file = "faculty_details.txt"  # Output text file name
    extract_faculty_details(input_html, output_file)
    print(f"Faculty details have been extracted and saved to {output_file}")
