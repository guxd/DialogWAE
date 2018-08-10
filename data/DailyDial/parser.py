__author__ = "Sanghoon Kang"
# DailyDialogue-Parser
# Parser for DailyDialogue Dataset

# Dataset Source: http://yanran.li/dailydialog

## Usage

###
# python3 parser.py -i <input_dir> -o <output_dir>
###

import os, sys, getopt, gzip

def parse_data(in_dir, out_dir):

    # Finding files
    if in_dir.endswith('train'):
        dial_dir = os.path.join(in_dir, 'dialogues_train.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_train.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_train.txt')
    elif in_dir.endswith('validation'):
        dial_dir = os.path.join(in_dir, 'dialogues_validation.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_validation.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_validation.txt')
    elif in_dir.endswith('test'):
        dial_dir = os.path.join(in_dir, 'dialogues_test.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_test.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_test.txt')
    else:
        print("Cannot find directory")
        sys.exit()

    out_dial_dir = os.path.join(out_dir, 'dial.txt.gz')
    out_emo_dir = os.path.join(out_dir, 'emo.txt.gz')
    out_act_dir = os.path.join(out_dir, 'act.txt.gz')

    # Open files
    in_dial = open(dial_dir, 'r')
    in_emo = open(emo_dir, 'r')
    in_act = open(act_dir, 'r')

    out_dial = gzip.open(out_dial_dir, 'w')
    out_emo = gzip.open(out_emo_dir, 'w')
    out_act = gzip.open(out_act_dir, 'w')


    for line_count, (line_dial, line_emo, line_act) in enumerate(zip(in_dial, in_emo, in_act)):
        seqs = line_dial.split('__eou__')
        seqs = seqs[:-1]

        emos = line_emo.split(' ')
        emos = emos[:-1]

        acts = line_act.split(' ')
        acts = acts[:-1]
        
        seq_count = 0
        seq_len = len(seqs)
        emo_len = len(emos)
        act_len = len(acts)
    
        if seq_len != emo_len or seq_len != act_len:
            print("Different turns btw dialogue & emotion & acttion! ", line_count+1, seq_len, emo_len, act_len)
            sys.exit()

        for seq, emo, act in zip(seqs, emos, acts):

            # Get rid of the blanks at the start & end of each turns
            if seq[0] == ' ':
                seq = seq[1:]
            if seq[-1] == ' ':
                seq = seq[:-1]

            out_dial.write(seq.encode('utf-8'))
            out_dial.write('\n'.encode('utf-8'))
            out_emo.write(emo.encode('utf-8'))
            out_emo.write('\n'.encode('utf-8'))
            out_act.write(act.encode('utf-8'))
            out_act.write('\n'.encode('utf-8'))

            if seq_count != 0 and seq_count != seq_len-1:
                out_dial.write(seq.encode('utf-8'))
                out_dial.write('\n'.encode('utf-8'))
                out_emo.write(emo.encode('utf-8'))
                out_emo.write('\n'.encode('utf-8'))
                out_act.write(act.encode('utf-8'))
                out_act.write('\n'.encode('utf-8'))

            seq_count += 1       

    in_dial.close()
    in_emo.close()
    in_act.close()
    out_dial.close()
    out_emo.close()
    out_act.close()

def main(argv):

    in_dir = ''
    out_dir = ''

    try:
        opts, args = getopt.getopt(argv,"h:i:o:",["in_dir=","out_dir="])
    except getopt.GetoptError:
        print("python3 parser.py -i <in_dir> -o <out_dir>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("python3 parser.py -i <in_dir> -o <out_dir>")
            sys.exit()
        elif opt in ("-i", "--in_dir"):
            in_dir = arg
        elif opt in ("-o", "--out_dir"):
            out_dir = arg

    print("Input directory : ", in_dir)
    print("Ouptut directory: ", out_dir)

    parse_data(in_dir, out_dir)

if __name__ == '__main__':
    main(sys.argv[1:])
