CHUNK_SIZE=500
CHUNK_OVERLAP=64
MIN_CHUNK_SIZE=40
SEPARATORS=["\n\n","\n"," ",""]

def estimate_tokens(text:str)->int:
    return max(1, len(text)//4)

def _split_on(text:str, separator:str)->list[str]:
    if not separator:
        return list(text)
    pieces=text.split(separator)
    result=[]
    for i,piece in enumerate(pieces):
        if piece:
            result.append(piece+separator if i<len(pieces)-1 else piece)
    return [p for p in result if p.strip()]

def _merge_into_chunks(pieces: list[str], chunk_size: int, overlap:int)->list[str]:
    chunks=[]
    current=""
    current_tokens=0
    for piece in pieces:
        piece_tokens= estimate_tokens(piece)
        if current_tokens+piece_tokens>chunk_size and current:
            chunks.append(current.strip())
            overlap_text=current[-(overlap*4):]
            current=overlap_text+piece
            current_tokens=estimate_tokens(current)
        else:
            current+=piece
            current_tokens+=piece_tokens
    if current.strip():
        chunks.append(current.strip())
    return chunks

def _recursive_split(text:str, sepearators: list[str], chunk_size:int)->list[str]:
    separator=sepearators[0]
    remaining_separators=sepearators[1:]
    pieces=_split_on(text,separator)
    good_pieces=[] #pieces that are small enough to use directly
    final_pieces=[] # final output

    for piece in pieces:
        if estimate_tokens(piece)<=chunk_size:
            good_pieces.append(piece)
        else:
            if good_pieces:
                final_pieces.extend(good_pieces)
                good_pieces=[]
            if remaining_separators:
                final_pieces.extend(_recursive_split(piece,remaining_separators,chunk_size))
            else:
                step=chunk_size*4
                for i in range(0,len(piece),step):
                    final_pieces.append(piece[i:i+step])
    if good_pieces:
        final_pieces.extend(good_pieces)
    return final_pieces

def chunk_document(text:str, source_name:str)-> list[dict]:
    cleaned=_clean(text)
    if estimate_tokens(text)<= CHUNK_SIZE:
        return [{
            "text":cleaned,
            "source":source_name,
            "chunk_index":0,
            "token_count":estimate_tokens(cleaned),
            "metadata":{"original_length":len(text)},
        }]
    raw_pieces=_recursive_split(cleaned, SEPARATORS , CHUNK_SIZE)
    merged=_merge_into_chunks(raw_pieces, CHUNK_SIZE, CHUNK_OVERLAP)
    result=[]
    for i, chunk_text in enumerate(merged):
        token_count=estimate_tokens(chunk_text)
        if token_count >=MIN_CHUNK_SIZE:
            result.append({
                "text":chunk_text,
                "source":source_name,
                "chunk_index":i,
                "token_count":token_count, 
            })
    return result

def _clean(text:str)->str:
    import re
    text=text.replace("\r\n", "\n").replace("\r","\n")
    text=text.replace("\t"," ")
    text=re.sub(r"\n{4,}","\n\n\n", text)
    text=re.sub(r" {3,}", " ",text)

    return text.strip()