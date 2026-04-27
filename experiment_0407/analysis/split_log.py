import re
import os

def parse_time(line):
    match = re.match(r'(\d{2}):(\d{2}):(\d{2}):(\d{3}):(\d{3})', line)
    if match:
        h, m, s, ms1, ms2 = match.groups()
        total_ms = (int(h)*3600 + int(m)*60 + int(s)) * 1000 + int(ms1)
        return total_ms
    return None

def split_log_by_gaps(filepath, gap_threshold_ms=3000):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    adv_lines = []
    for lineno, line in enumerate(all_lines):
        if '[ADV]' in line:
            t = parse_time(line)
            if t is not None:
                adv_lines.append((lineno, t))
    
    split_points = []
    for i in range(1, len(adv_lines)):
        prev_lineno, prev_t = adv_lines[i-1]
        curr_lineno, curr_t = adv_lines[i]
        gap = curr_t - prev_t
        if gap > gap_threshold_ms:
            split_points.append(curr_lineno)
            print(f"전환 감지: 라인 {curr_lineno+1}, gap {gap/1000:.1f}초")
    
    print(f"\n총 {len(split_points)}개 전환 감지 → {len(split_points)+1}개 파일 생성\n")
    
    boundaries = [0] + split_points + [len(all_lines)]
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        
        out_filename = os.path.join(script_dir, f"{base_name}_{i+1}.txt")
        with open(out_filename, 'w', encoding='utf-8') as f:
            f.writelines(all_lines[start:end])
        
        print(f"파일 {i+1}: 라인 {start+1}~{end} → {os.path.basename(out_filename)}")

split_log_by_gaps("log_20260407_231508.txt", gap_threshold_ms=3000)