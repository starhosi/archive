from numpy import array

def split_sequence3(sequence,n_steps):
    X,y = list(),list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix, :],sequence[end_ix,:]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])