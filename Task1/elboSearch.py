import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wsd_alert_system import BertWSD

def plot_results(all_stats, thresholds, global_f1_scores):
    plt.figure(figsize=(12, 8))
    
    # Plot individual words
    for word, stats in all_stats.items():
        # stats is list of (thresh, prec, rec, f1)
        f1_scores = [s[3] for s in stats]
        plt.plot(thresholds, f1_scores, label=f"{word}", alpha=0.5, linestyle='--')
        
    # Plot global average
    plt.plot(thresholds, global_f1_scores, label="Global Average", color='black', linewidth=3)
    
    # Find max of global average for annotation
    max_f1 = max(global_f1_scores)
    max_thresh = thresholds[np.argmax(global_f1_scores)]
    
    plt.annotate(f'Optimal Threshold: {max_thresh:.2f}\nF1: {max_f1:.4f}', 
                 xy=(max_thresh, max_f1), 
                 xytext=(max_thresh, max_f1 + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center')

    plt.title("F1 Score vs. Similarity Threshold (Elbow Method)")
    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("F1 Score")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.xticks(np.arange(0.1, 1.0, 0.05))
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_elbow_plot.png")
    plt.savefig(output_file)
    print(f"\nPlot saved to: {output_file}")

def get_labeled_data():
    # Format: (text, is_target_sense)
    # Target: "bank" as Financial Institution
    data_bank = [
        # Positive (Financial)
        ("I went to the bank to deposit my check.", True),
        ("The bank approved my loan application yesterday.", True),
        ("She works as a teller at the local bank.", True),
        ("The central bank raised interest rates.", True),
        ("He keeps his savings in a secure bank account.", True),
        ("Investment banks played a role in the crisis.", True),
        ("I need to find an ATM from my bank.", True),
        ("The bank manager was very helpful.", True),
        ("Online banking has made life easier.", True),
        ("The robbery at the city bank was thwarted.", True),
        ("I need to stop by the bank to withdraw some cash.", True),
        ("The bank denied his mortgage application due to poor credit.", True),
        ("She works as a manager at the First National Bank.", True),
        ("Please send the wire transfer to my bank account.", True),
        ("He keeps his valuables in a safety deposit box at the bank.", True),
        ("The bank robbery made headlines across the country.", True),
        ("The bank charges a fee for international transactions.", True),
        ("I forgot my PIN at the bank ATM.", True),
        ("The investment bank underwrote the IPO.", True),
        ("They opened a joint bank account after getting married.", True),
        ("The bank teller counted the money twice.", True),
        ("A run on the bank caused panic among depositors.", True),
        ("The World Bank provides loans to developing countries.", True),
        ("My paycheck is directly deposited into the bank.", True),
        ("The bank's vault is secured by a time lock.", True),
        ("He cashed the check at the issuing bank.", True),
        ("The bank statement shows a discrepancy.", True),
        ("Commercial banks offer loans to businesses.", True),
        ("The bank merger was approved by regulators.", True),
        ("She took out a loan from the bank to buy a car.", True),
        ("The bank is open from 9 to 5.", True),
        ("Swiss banks are known for their privacy.", True),
        ("The bank balance was lower than expected.", True),
        ("He sued the bank for predatory lending.", True),
        ("The bank draft was lost in the mail.", True),
        ("Merchant banks deal with international finance.", True),
        ("The bank holiday delayed the transaction.", True),
        
        # Negative (River/Geographical)
        ("The river bank was muddy after the rain.", False),
        ("He sat on the bank of the river and fished.", False),
        ("They had a picnic on the grassy bank.", False),
        ("The boat drifted towards the steep bank.", False),
        ("Erosion is damaging the river bank.", False),
        ("We sat on the river bank and watched the sunset.", False),
        ("The boat ran aground on the sandy bank.", False),
        ("Heavy rains caused the river to overflow its banks.", False),
        ("He fished from the bank of the lake.", False),
        ("She planted flowers along the grassy bank.", False),
        ("The snow bank was piled high against the door.", False),
        ("The otter slid down the muddy bank.", False),
        ("They built a levee to reinforce the river bank.", False),
        ("The canoe was pulled up onto the bank.", False),
        ("The West Bank is a region in the Middle East.", False),
        ("The Left Bank of Paris is famous for its artists.", False),
        
        # Negative (Verb/Movement)
        ("The plane banked sharply to the left.", False),
        ("The car banked around the corner.", False),
        ("Clouds banked up against the mountains.", False),
        ("The plane banked steeply to avoid the mountain.", False),
        ("The car banked around the sharp curve of the track.", False),
        ("The pilot banked the aircraft to the right.", False),
        ("The race car banked on the turn.", False),
        ("He banked the fire to keep it burning overnight.", False),
        
        # Negative (Metaphorical/Other)
        ("Data banks are essential for modern computing.", False),
        ("I wouldn't bank on him showing up on time.", False),
        ("The blood bank is running low on supplies.", False),
        ("He has a vast memory bank of trivia.", False),
        ("A bank of clouds appeared on the horizon.", False),
        ("A fog bank rolled in from the sea.", False),
        ("He has a bank of monitors on his desk.", False),
        ("Don't bank on the weather being nice tomorrow.", False),
        ("The data bank stores millions of records.", False),
        ("The blood bank is asking for donations.", False),
        ("The memory bank of the computer was corrupted.", False),
        ("A bank of elevators services the skyscraper.", False),
        ("The switch bank controls the stage lighting.", False),
        ("A bank of batteries powers the backup system.", False),
        ("The sperm bank preserves genetic material.", False),
        ("The food bank distributes meals to the needy.", False),
        ("The coin rolled down the steep bank.", False),
        ("A bank of lights illuminated the stadium.", False)
    ]
    
    # Target: "cell" as Prison Room
    data_cell = [
        # Positive (Prison)
        ("The prisoner was taken back to his cell.", True),
        ("The guard locked the cell door.", True),
        ("He spent ten years in a prison cell.", True),
        ("The monk retreated to his small cell for prayer.", True),
        ("The inmate paced around his small cell.", True),
        ("They found contraband hidden in the cell.", True),
        ("The prisoner paced back and forth in his cell.", True),
        ("The guard locked the heavy iron door of the cell.", True),
        ("He was placed in a holding cell overnight.", True),
        ("The inmate spent 23 hours a day in his cell.", True),
        ("They found a shank hidden under the mattress in the cell.", True),
        ("The monk lived in a simple cell with only a bed and a desk.", True),
        ("She was transferred to a maximum-security cell.", True),
        ("The cell was cold and damp.", True),
        ("He wrote letters from his prison cell.", True),
        ("The escapees dug a tunnel out of their cell.", True),
        ("Solitary confinement means being in a cell alone.", True),
        ("The cell block was on lockdown after the riot.", True),
        ("The nun retired to her cell for contemplation.", True),
        ("He was released from his cell for good behavior.", True),
        ("The walls of the cell were covered in graffiti.", True),
        ("A padded cell is used for patients who might hurt themselves.", True),
        ("The jail cell had a small barred window.", True),
        ("He shared a cell with a notorious gangster.", True),
        ("The cell inspection revealed contraband.", True),
        ("The prisoner was escorted back to his cell.", True),
        ("The hermit's cell was a cave in the mountains.", True),
        ("The police put the drunk driver in the drunk tank cell.", True),
        ("The cell measured only six by eight feet.", True),
        ("He heard the keys jingle outside his cell.", True),
        ("The death row cell was near the execution chamber.", True),
        ("The monastery has many small cells for the brothers.", True),
        ("She was thrown into a dark dungeon cell.", True),
        ("The cell door slammed shut with a loud clang.", True),
        ("He cleaned his cell every morning.", True),
        ("The warden visited the prisoner in his cell.", True),
        
        # Negative (Biology)
        ("Red blood cells carry oxygen.", False),
        ("The plant cell has a rigid wall.", False),
        ("Stem cell research is controversial.", False),
        ("The cancer cells multiplied rapidly.", False),
        ("Red blood cells carry oxygen to the body.", False),
        ("The plant cell has a rigid cell wall.", False),
        ("Stem cell therapy shows promise for treating diseases.", False),
        ("White blood cells fight infection.", False),
        ("The organism is made of a single cell.", False),
        ("Cell division occurs through mitosis.", False),
        ("Sickle cell anemia is a genetic disorder.", False),
        ("The brain cells die without oxygen.", False),
        ("The cell membrane controls what enters the cell.", False),
        
        # Negative (Technology/Phone)
        ("She checked her cell for missed calls.", False),
        ("The cell tower was damaged in the storm.", False),
        ("My cell battery is dead.", False),
        ("I left my cell phone in the car.", False),
        ("Solar cells convert sunlight into electricity.", False),
        ("Call me on my cell if you need anything.", False),
        ("The battery cell needs to be replaced.", False),
        ("The fuel cell powers the electric vehicle.", False),
        ("The cell tower provides coverage for the area.", False),
        ("My cell reception is terrible here.", False),
        ("The photovoltaic cell efficiency has improved.", False),
        ("The dry cell battery is common in electronics.", False),
        ("She bought a new case for her cell.", False),
        
        # Negative (Group)
        ("The terrorist cell was planning an attack.", False),
        ("A sleeper cell was activated.", False),
        ("The terrorist cell was planning a coordinated attack.", False),
        ("A sleeper cell was activated in the city.", False),
        ("The radical cell met in secret.", False),
        ("The undercover agent infiltrated the criminal cell.", False),
        
        # Negative (Spreadsheet/Other)
        ("Enter the formula in cell B4.", False),
        ("The spreadsheet cell contains an error.", False),
        ("The spreadsheet cell contains a formula.", False),
        ("The animation cell was hand-painted.", False),
        ("The storm cell produced a tornado.", False),
        ("He highlighted the cell in Excel.", False),
        ("Format the cell to display currency.", False),
        ("The bee comb is made of hexagonal cells.", False),
        ("The thunder cell moved across the plains.", False)
    ]

    # Target: "bat" as Animal
    data_bat = [
        # Positive (Animal)
        ("The bat flew out of the cave at dusk.", True),
        ("Bats are the only mammals capable of true flight.", True),
        ("The fruit bat hangs upside down from the tree.", True),
        ("We saw a bat swooping for insects.", True),
        ("The bat uses echolocation to navigate.", True),
        ("Vampire bats feed on blood.", True),
        ("The bat colony roosts in the old barn.", True),
        ("A bat fluttered around the street lamp.", True),
        ("The bat's wingspan was impressive.", True),
        ("He is afraid of bats.", True),
        
        # Negative (Sports)
        ("He swung the baseball bat with all his might.", False),
        ("The cricket bat is made of willow wood.", False),
        ("She bought a new softball bat.", False),
        ("The player dropped his bat and ran to first base.", False),
        ("The bat cracked when he hit the ball.", False),
        ("He is the best bat in the lineup.", False),
        
        # Negative (Verb/Idiom)
        ("She didn't bat an eye when she heard the news.", False),
        ("He batted the ball away with his hand.", False),
        ("The cat batted at the toy mouse.", False),
        ("She batted her eyelashes at him.", False)
    ]

    # Target: "apple" as Fruit
    data_apple = [
        # Positive (Fruit)
        ("She ate a crisp red apple.", True),
        ("The apple pie smelled delicious.", True),
        ("He picked a ripe apple from the tree.", True),
        ("An apple a day keeps the doctor away.", True),
        ("The apple juice was sweet and refreshing.", True),
        ("She peeled the apple with a knife.", True),
        ("The apple core was thrown in the trash.", True),
        ("Granny Smith is a type of tart apple.", True),
        ("The basket was full of fresh apples.", True),
        ("He took a bite of the juicy apple.", True),
        
        # Negative (Company/Tech)
        ("Apple released the new iPhone yesterday.", False),
        ("He works as a software engineer at Apple.", False),
        ("The Apple Store was crowded with customers.", False),
        ("Apple's stock price rose significantly.", False),
        ("She uses an Apple Watch to track her fitness.", False),
        ("The lawsuit against Apple was settled.", False),
        
        # Negative (Place)
        ("The Big Apple is a nickname for New York City.", False),
        ("They visited the Big Apple for vacation.", False)
    ]

    # Target: "date" as Calendar/Time
    data_date = [
        # Positive (Calendar)
        ("What is today's date?", True),
        ("Please write the date on the top of the page.", True),
        ("The expiration date is printed on the bottle.", True),
        ("Save the date for our wedding.", True),
        ("The date of the meeting has been changed.", True),
        ("He was born on a date in July.", True),
        ("Check the date on the milk carton.", True),
        ("The release date for the movie is next month.", True),
        ("Do you have a date in mind for the party?", True),
        ("The due date for the assignment is Friday.", True),
        
        # Negative (Fruit)
        ("The sticky date pudding was delicious.", False),
        ("Dates are a sweet fruit from the palm tree.", False),
        ("He ate a handful of dried dates.", False),
        ("The date palm grows in arid climates.", False),
        ("She added chopped dates to the oatmeal.", False),
        
        # Negative (Social)
        ("They went on a romantic date to the movies.", False),
        ("He asked her out on a date.", False),
        ("She has a blind date tonight.", False),
        ("The date went well and they planned to meet again.", False),
        ("He is dating a girl from work.", False)
    ]
    
    return {
        "bank": data_bank,
        "cell": data_cell,
        "bat": data_bat,
        "apple": data_apple,
        "date": data_date
    }

def evaluate_thresholds(wsd_system, data, target_word, definition, log_file=None):
    msg = f"\nEvaluating for target word: '{target_word}'"
    print(msg)
    if log_file: log_file.write(msg + "\n")
    
    msg = f"Definition: {definition}"
    print(msg)
    if log_file: log_file.write(msg + "\n")
    
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    
    # Run with threshold 0 to get all scores
    results = wsd_system.scan_dataset(texts, target_word, definition, threshold=0.0)
    
    # Map scores to doc_ids
    scores = np.zeros(len(texts))
    
    for res in results:
        doc_id = res['doc_id']
        if scores[doc_id] < res['similarity']:
            scores[doc_id] = res['similarity']
            
    # Sweep thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_f1 = 0
    best_thresh = 0
    
    header = f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Accuracy':<10}"
    print(header)
    if log_file: log_file.write(header + "\n")
    
    separator = "-" * 60
    print(separator)
    if log_file: log_file.write(separator + "\n")
    
    stats = []
    
    for thresh in thresholds:
        predictions = (scores >= thresh)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        
        row = f"{thresh:<10.2f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {accuracy:<10.4f}"
        print(row)
        if log_file: log_file.write(row + "\n")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
        stats.append((thresh, precision, recall, f1))
            
    print(separator)
    if log_file: log_file.write(separator + "\n")
    
    footer = f"Best F1 Score: {best_f1:.4f} at Threshold: {best_thresh:.2f}"
    print(footer)
    if log_file: log_file.write(footer + "\n")
    
    return stats

def main():
    wsd = BertWSD()
    datasets = get_labeled_data()
    
    definitions = {
        "bank": "A financial institution that accepts deposits and channels the money into lending activities.",
        "cell": "A small room in which a prisoner is locked up.",
        "bat": "A mainly nocturnal mammal capable of sustained flight, with membranous wings.",
        "apple": "The round fruit of a tree of the rose family, which typically has thin red or green skin and crisp flesh.",
        "date": "The day of the month or year as specified by a number."
    }
    
    all_stats = {}
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_results.txt")
    
    with open(output_path, "w") as log_file:
        print(f"Writing results to {output_path}")
        
        for word, data in datasets.items():
            stats = evaluate_thresholds(wsd, data, word, definitions[word], log_file=log_file)
            all_stats[word] = stats

        # Aggregate results to find global best threshold
        header = "\n" + "="*60 + "\nGLOBAL OPTIMIZATION\n" + "="*60
        print(header)
        log_file.write(header + "\n")
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_global_f1 = 0
        best_global_thresh = 0
        
        header_row = f"{'Threshold':<10} | {'Avg F1':<10}"
        print(header_row)
        log_file.write(header_row + "\n")
        
        separator = "-" * 25
        print(separator)
        log_file.write(separator + "\n")
        
        global_f1_scores = []
        
        for i, thresh in enumerate(thresholds):
            f1_scores = []
            for word in datasets.keys():
                f1 = all_stats[word][i][3]
                f1_scores.append(f1)
            
            avg_f1 = np.mean(f1_scores)
            global_f1_scores.append(avg_f1)
            
            row = f"{thresh:<10.2f} | {avg_f1:<10.4f}"
            print(row)
            log_file.write(row + "\n")
            
            if avg_f1 > best_global_f1:
                best_global_f1 = avg_f1
                best_global_thresh = thresh
                
        print(separator)
        log_file.write(separator + "\n")
        
        footer = f"Best Global Average F1: {best_global_f1:.4f} at Threshold: {best_global_thresh:.2f}"
        print(footer)
        log_file.write(footer + "\n")

    # Plot results
    plot_results(all_stats, thresholds, global_f1_scores)

if __name__ == "__main__":
    main()
