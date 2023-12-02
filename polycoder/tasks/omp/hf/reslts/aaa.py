from prettytable import PrettyTable
import re

def find_clause(line, clause):
    if clause not in line:
        return ''
    
    new_line = re.sub(r'(\w)\s*\(', r'\1(', line)

    clause_idx = new_line.find(clause)

    if len(new_line)<clause_idx+len(clause) or new_line[clause_idx+len(clause)] != '(':
        return f'{clause} ( )'
    
    paren_idx = new_line[clause_idx:].find(')')
    if paren_idx == -1:
        return f'{new_line[clause_idx:]} )'

    return new_line[clause_idx:clause_idx+paren_idx+1]


pred_table = PrettyTable()
pred_table.field_names = ["Label", "Pred"]
pred_table.align["Label"] = "l"
pred_table.align["Pred"] = "l"


with open('final_poly_parallel_bpe_results.log', 'r') as f, open('final_poly_parallel_bpe_results_clean.log', 'w') as out:
    lines = f.readlines()[3:-1]
    for line in lines:
        
        splitted_line = line.split('|')
        label, pred = splitted_line[1], splitted_line[2]
        pred_table.add_row([label, f"parallel for {find_clause(pred, 'private')} {find_clause(pred, 'reduction')}"])

    out.write(str(pred_table))