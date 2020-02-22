# fast.jl
  
function reduce_tokens(token2idx::Dict{Any, Any}, token_counts::Dict{Any, Any}, idx_array::Array{Int, 1}, threshold::Int)

    new_t2i = Dict("<unk>"=>0)
    new_idx_array = copy(idx_array)

    for (token, _) in token2idx

        # get the old index and the token count
        old_idx = token2idx[token]
        count = token_counts[token]

        # new index: merged with unknown if below threshold
        above_threshold = count >= threshold
        if above_threshold
            new_idx = length(Set(values(new_t2i)))
        else
            new_idx = 0
        end

        # change the t2i and idx_array
        new_t2i[token] = new_idx
        new_idx_array[idx_array.==old_idx] .= new_idx

        # remove counts
        if !above_threshold
            if token != "<unk>"
                token_counts[token] = 0
            end
            token_counts["<unk>"] += count
        end

    end

    return new_t2i, token_counts, idx_array
end

