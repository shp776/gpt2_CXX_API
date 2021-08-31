// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

// implement code
// g++ -o Final_CXX_Api_gpt Final_CXX_Api_gpt.cpp simdjson.cpp ctre-unicode.hpp -I ./include/onnxruntime/core/session -lonnxruntime -std=c++17
// ./Final_CXX_Api_gpt -t "Here is some text to encode Hello World" -n 5
// ./Final_CXX_Api_gpt -t "Here is" -n 15

#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>

// post.cpp header
#include <string_view>
#include <iostream>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <random>
#include <ctime>    // for time checking

#include "ctre-unicode.hpp"
#include "simdjson.h"
#include "cxxopts.hpp"  // option_parser_library


using namespace std;
static constexpr size_t BATCH_SIZE = 1;


// topk 알고리즘
vector<float> topK(vector<float> &a, int k)
{
    if(k!=0) {  // k=0일때 return logits 그대로

    unordered_map<float, int> u;        // 뒤에값이 key
    int n = a.size();
    int i;

    for (i = 0; i < n; i++)
    {
        u[a[i]]++;  //  mapping..
    }

    vector<pair<float, int>> v(u.begin(), u.end());

    sort(v.begin(), v.end(), [](pair<float, int> x, pair<float, int> y) {
        return x.first > y.first;
    });

    vector<float> r;
    for (i = 0; i < k; i++)
    {
        r.push_back(v[i].first);
    }

    // cout <<"\n\n unordered_map size is " << u.size() << "\n";    // for checking

    float min = *min_element(r.begin(), r.end());

    // torch.where 
    // replace_if(
    //     r.begin(), r.end(), [](const int &i) { return i < min; }, (-1*exp(10)));

    replace_if( a.begin(), a.end(), bind2nd(std::less<float>(), min), (-1*exp(10)) );  

    return a;
    }

    else return a;
}

//  topk 알고리즘 후 torch.multinomial
template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}


//  iterator 반환 view 구조 정의
template <typename T>
struct view {      // vector iterator를 반환하도록 begin()과 end() 재정의.
    typename std::vector<T>::iterator _start;
    typename std::vector<T>::iterator _end;

    auto begin() const {
        return _start;
    }
    auto end() const {
        return _end;
    }
};

//  hash_combine for hash_table
template <class T>
inline void hash_combine(size_t& seed, const T& v)
{
    hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

unordered_map<char, string> bytes_to_unicode() {
    // Because I have no idea what I am doing the code_map was copy pasted from bytes_to_unicode()
    // 내가 무엇을 하고 있는지 모르기 때문에 code_map은 bytes_to_unicode()에서 복사하여 붙여넣었습니다.
    // definetely not crossplatform, but works on POSIX maybe
    static unordered_map<char, string> code_map = {{33, "!"},{34, "\""},{35, "#"},{36, "$"},{37, "%"},{38, "&"},{39, "\'"},{40, "("},{41, ")"},{42, "*"},{43, "+"},{44, ","},{45, "-"},{46, "."},{47, "/"},{48, "0"},{49, "1"},{50, "2"},{51, "3"},{52, "4"},{53, "5"},{54, "6"},{55, "7"},{56, "8"},{57, "9"},{58, ":"},{59, ";"},{60, "<"},{61, "="},{62, ">"},{63, "?"},{64, "@"},{65, "A"},{66, "B"},{67, "C"},{68, "D"},{69, "E"},{70, "F"},{71, "G"},{72, "H"},{73, "I"},{74, "J"},{75, "K"},{76, "L"},{77, "M"},{78, "N"},{79, "O"},{80, "P"},{81, "Q"},{82, "R"},{83, "S"},{84, "T"},{85, "U"},{86, "V"},{87, "W"},{88, "X"},{89, "Y"},{90, "Z"},{91, "["},{92, "\\"},{93, "]"},{94, "^"},{95, "_"},{96, "`"},{97, "a"},{98, "b"},{99, "c"},{100, "d"},{101, "e"},{102, "f"},{103, "g"},{104, "h"},{105, "i"},{106, "j"},{107, "k"},{108, "l"},{109, "m"},{110, "n"},{111, "o"},{112, "p"},{113, "q"},{114, "r"},{115, "s"},{116, "t"},{117, "u"},{118, "v"},{119, "w"},{120, "x"},{121, "y"},{122, "z"},{123, "{"},{124, "|"},{125, "}"},{126, "~"},{161, "¡"},{162, "¢"},{163, "£"},{164, "¤"},{165, "¥"},{166, "¦"},{167, "§"},{168, "¨"},{169, "©"},{170, "ª"},{171, "«"},{172, "¬"},{174, "®"},{175, "¯"},{176, "°"},{177, "±"},{178, "²"},{179, "³"},{180, "´"},{181, "µ"},{182, "¶"},{183, "·"},{184, "¸"},{185, "¹"},{186, "º"},{187, "»"},{188, "¼"},{189, "½"},{190, "¾"},{191, "¿"},{192, "À"},{193, "Á"},{194, "Â"},{195, "Ã"},{196, "Ä"},{197, "Å"},{198, "Æ"},{199, "Ç"},{200, "È"},{201, "É"},{202, "Ê"},{203, "Ë"},{204, "Ì"},{205, "Í"},{206, "Î"},{207, "Ï"},{208, "Ð"},{209, "Ñ"},{210, "Ò"},{211, "Ó"},{212, "Ô"},{213, "Õ"},{214, "Ö"},{215, "×"},{216, "Ø"},{217, "Ù"},{218, "Ú"},{219, "Û"},{220, "Ü"},{221, "Ý"},{222, "Þ"},{223, "ß"},{224, "à"},{225, "á"},{226, "â"},{227, "ã"},{228, "ä"},{229, "å"},{230, "æ"},{231, "ç"},{232, "è"},{233, "é"},{234, "ê"},{235, "ë"},{236, "ì"},{237, "í"},{238, "î"},{239, "ï"},{240, "ð"},{241, "ñ"},{242, "ò"},{243, "ó"},{244, "ô"},{245, "õ"},{246, "ö"},{247, "÷"},{248, "ø"},{249, "ù"},{250, "ú"},{251, "û"},{252, "ü"},{253, "ý"},{254, "þ"},{255, "ÿ"},{0, "Ā"},{1, "ā"},{2, "Ă"},{3, "ă"},{4, "Ą"},{5, "ą"},{6, "Ć"},{7, "ć"},{8, "Ĉ"},{9, "ĉ"},{10, "Ċ"},{11, "ċ"},{12, "Č"},{13, "č"},{14, "Ď"},{15, "ď"},{16, "Đ"},{17, "đ"},{18, "Ē"},{19, "ē"},{20, "Ĕ"},{21, "ĕ"},{22, "Ė"},{23, "ė"},{24, "Ę"},{25, "ę"},{26, "Ě"},{27, "ě"},{28, "Ĝ"},{29, "ĝ"},{30, "Ğ"},{31, "ğ"},{32, "Ġ"},{127, "ġ"},{128, "Ģ"},{129, "ģ"},{130, "Ĥ"},{131, "ĥ"},{132, "Ħ"},{133, "ħ"},{134, "Ĩ"},{135, "ĩ"},{136, "Ī"},{137, "ī"},{138, "Ĭ"},{139, "ĭ"},{140, "Į"},{141, "į"},{142, "İ"},{143, "ı"},{144, "Ĳ"},{145, "ĳ"},{146, "Ĵ"},{147, "ĵ"},{148, "Ķ"},{149, "ķ"},{150, "ĸ"},{151, "Ĺ"},{152, "ĺ"},{153, "Ļ"},{154, "ļ"},{155, "Ľ"},{156, "ľ"},{157, "Ŀ"},{158, "ŀ"},{159, "Ł"},{160, "ł"},{173, "Ń"}};
    return code_map;
}

unordered_map<string, char> unicode_to_bytes() {
    static unordered_map<string, char> code_map = {{"!", 33},{"\"", 34},{"#", 35},{"$", 36},{"%", 37},{"&", 38},{"\'", 39},{"(", 40},{")", 41},{"*", 42},{"+", 43},{",", 44},{"-", 45},{".", 46},{"/", 47},{"0", 48},{"1", 49},{"2", 50},{"3", 51},{"4", 52},{"5", 53},{"6", 54},{"7", 55},{"8", 56},{"9", 57},{":", 58},{";", 59},{"<", 60},{"=", 61},{">", 62},{"?", 63},{"@", 64},{"A", 65},{"B", 66},{"C", 67},{"D", 68},{"E", 69},{"F", 70},{"G", 71},{"H", 72},{"I", 73},{"J", 74},{"K", 75},{"L", 76},{"M", 77},{"N", 78},{"O", 79},{"P", 80},{"Q", 81},{"R", 82},{"S", 83},{"T", 84},{"U", 85},{"V", 86},{"W", 87},{"X", 88},{"Y", 89},{"Z", 90},{"[", 91},{"\\", 92},{"]", 93},{"^", 94},{"_", 95},{"`", 96},{"a", 97},{"b", 98},{"c", 99},{"d", 100},{"e", 101},{"f", 102},{"g", 103},{"h", 104},{"i", 105},{"j", 106},{"k", 107},{"l", 108},{"m", 109},{"n", 110},{"o", 111},{"p", 112},{"q", 113},{"r", 114},{"s", 115},{"t", 116},{"u", 117},{"v", 118},{"w", 119},{"x", 120},{"y", 121},{"z", 122},{"{", 123},{"|", 124},{"}", 125},{"~", 126},{"¡", 161},{"¢", 162},{"£", 163},{"¤", 164},{"¥", 165},{"¦", 166},{"§", 167},{"¨", 168},{"©", 169},{"ª", 170},{"«", 171},{"¬", 172},{"®", 174},{"¯", 175},{"°", 176},{"±", 177},{"²", 178},{"³", 179},{"´", 180},{"µ", 181},{"¶", 182},{"·", 183},{"¸", 184},{"¹", 185},{"º", 186},{"»", 187},{"¼", 188},{"½", 189},{"¾", 190},{"¿", 191},{"À", 192},{"Á", 193},{"Â", 194},{"Ã", 195},{"Ä", 196},{"Å", 197},{"Æ", 198},{"Ç", 199},{"È", 200},{"É", 201},{"Ê", 202},{"Ë", 203},{"Ì", 204},{"Í", 205},{"Î", 206},{"Ï", 207},{"Ð", 208},{"Ñ", 209},{"Ò", 210},{"Ó", 211},{"Ô", 212},{"Õ", 213},{"Ö", 214},{"×", 215},{"Ø", 216},{"Ù", 217},{"Ú", 218},{"Û", 219},{"Ü", 220},{"Ý", 221},{"Þ", 222},{"ß", 223},{"à", 224},{"á", 225},{"â", 226},{"ã", 227},{"ä", 228},{"å", 229},{"æ", 230},{"ç", 231},{"è", 232},{"é", 233},{"ê", 234},{"ë", 235},{"ì", 236},{"í", 237},{"î", 238},{"ï", 239},{"ð", 240},{"ñ", 241},{"ò", 242},{"ó", 243},{"ô", 244},{"õ", 245},{"ö", 246},{"÷", 247},{"ø", 248},{"ù", 249},{"ú", 250},{"û", 251},{"ü", 252},{"ý", 253},{"þ", 254},{"ÿ", 255},{"Ā", 0},{"ā", 1},{"Ă", 2},{"ă", 3},{"Ą", 4},{"ą", 5},{"Ć", 6},{"ć", 7},{"Ĉ", 8},{"ĉ", 9},{"Ċ", 10},{"ċ", 11},{"Č", 12},{"č", 13},{"Ď", 14},{"ď", 15},{"Đ", 16},{"đ", 17},{"Ē", 18},{"ē", 19},{"Ĕ", 20},{"ĕ", 21},{"Ė", 22},{"ė", 23},{"Ę", 24},{"ę", 25},{"Ě", 26},{"ě", 27},{"Ĝ", 28},{"ĝ", 29},{"Ğ", 30},{"ğ", 31},{"Ġ", 32},{"ġ", 127},{"Ģ", 128},{"ģ", 129},{"Ĥ", 130},{"ĥ", 131},{"Ħ", 132},{"ħ", 133},{"Ĩ", 134},{"ĩ", 135},{"Ī", 136},{"ī", 137},{"Ĭ", 138},{"ĭ", 139},{"Į", 140},{"į", 141},{"İ", 142},{"ı", 143},{"Ĳ", 144},{"ĳ", 145},{"Ĵ", 146},{"ĵ", 147},{"Ķ", 148},{"ķ", 149},{"ĸ", 150},{"Ĺ", 151},{"ĺ", 152},{"Ļ", 153},{"ļ", 154},{"Ľ", 155},{"ľ", 156},{"Ŀ", 157},{"ŀ", 158},{"Ł", 159},{"ł", 160},{"Ń", 173}};
    return code_map;
}

class GPT2Tokenizer {

    struct PairHash
    {
        size_t operator()(const pair<string, string>& p) const noexcept
        {
            size_t seed = 0;
            hash_combine(seed, p.first);
            hash_combine(seed, p.second);
            return seed;
        }
    };

    using BPE = pair<string, string>;
    using BPERanks = unordered_map<BPE, size_t, PairHash>;
    using Encoder = unordered_map<string, int64_t>;
    using Decoder = unordered_map<int64_t, string>;

    static constexpr string_view pattern {"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"};


public:

    static optional<GPT2Tokenizer> load(string_view vocab_file, string_view merges_file);

    vector<int64_t> encode(const string&);
    string decode(const vector<int64_t>&);
    vector<string> tokenize(const string&);

    size_t vocab_size() const noexcept { return m_encoder.size(); }
    
// protected:

    GPT2Tokenizer() = default;

    BPERanks m_bpe_ranks;
    Encoder m_encoder;
    Decoder m_decoder;
    unordered_map<char, string> m_byte_encoder;
    unordered_map<string, char> m_byte_decoder;

private:
    vector<string> bpe(const string& token);
};

    optional<GPT2Tokenizer> GPT2Tokenizer::load(string_view vocab_file, string_view merges_file) {

    // load merges file
    ifstream merges_file_stream;
    // assuming null-terminated string
    // merges_file_stream.open(merges_file.data());
    merges_file_stream.open("vocab.bpe");

    if (!merges_file_stream.good()) {
        return nullopt;
    }


    BPERanks bpe_ranks;

    for (struct{string line; size_t i{0};} it; getline(merges_file_stream, it.line); ++it.i) {        
            const size_t split_point = it.line.find(' ');
            pair<string, string> p{{it.line.begin(), it.line.begin()+split_point},
                                                {it.line.begin() + split_point + 1, it.line.end()}};
            bpe_ranks.emplace(move(p), it.i);
        }

    simdjson::dom::parser parser;
    simdjson::dom::object object;
    // assuming null-terminated string
    simdjson::dom::element doc = parser.load(vocab_file.data());
    // simdjson::dom::element doc = parser.load("encoder.json");

    auto error = doc.get(object);
    if (error) { 
        return nullopt; 
    }

    Encoder encoder;
    Decoder decoder;

    for (const auto& [key, value] : object) {
        encoder.emplace(key, value);
        decoder.emplace(value, key);
    }

    auto result = GPT2Tokenizer();
    result.m_bpe_ranks = move(bpe_ranks);
    result.m_encoder = move(encoder);
    result.m_decoder = move(decoder);
    result.m_byte_encoder = bytes_to_unicode();
    result.m_byte_decoder = unicode_to_bytes();

    return result;
}

template <typename T>
T unwrap(optional<T>&& value, const string& error_msg) {      // value 없을 시, error 출력
    if (value.has_value()) {
        return value.value();
    }
    else {
        throw runtime_error(error_msg);
    }
} 

size_t codepoint_length(const char c) {
    if((c & 0xf8) == 0xf0) return 4; // 4-byte unicode
    else if((c & 0xf0) == 0xe0) return 3; // 3-byte unicode
    else if((c & 0xe0) == 0xc0) return 2;
    else return 1;
}

vector<string> GPT2Tokenizer::bpe(const string& token) {

    vector<BPERanks::const_iterator> ranks;
    vector<string> word;
    ranks.reserve(token.size()-1);
    word.reserve(token.size());            

    // this essentially avoids having literal spaces ' ' in a string
    // at the same time we fetch the ranks of the bigrams
    {
        size_t i = 0;
        while (true) { // get_pairs role
            int length = codepoint_length(token[i]);
            int next_length = codepoint_length(token[i+length]);
            ranks.push_back(
                m_bpe_ranks.find({token.substr(i,length), token.substr(i+length,next_length)})
            );
            word.push_back(token.substr(i,length));
            i+=length;
            if (i >= token.size()) break;
            if (i+next_length >= token.size()) {
                word.emplace_back(token.substr(i,next_length));
                break;
            }
        }
    }
                                                // **********
    while (true) { 
        const auto bigram = min_element(ranks.begin(), ranks.end(),
                                             [this](const auto& lhs, const auto& rhs) -> bool {                 
                                                 if (lhs == m_bpe_ranks.end() && rhs == m_bpe_ranks.end()) {    
                                                     return false;
                                                 }
                                                 else if (lhs == m_bpe_ranks.end() || rhs == m_bpe_ranks.end()) {
                                                     return (lhs != m_bpe_ranks.end());
                                                 }
                                                 else {
                                                     return lhs->second < rhs->second;
                                                 }
                                             });
        if (*bigram == m_bpe_ranks.end()) {
            // could not find any matches in ranks
            break;
        }
        const auto [first, second] = (*bigram)->first;
        vector<string> new_word;

        size_t i = 0;
        while (i < word.size()) {
            const auto wordIterator = find(word.begin() + i, word.end(), first);
            if (wordIterator == word.end()) {
                copy(word.begin() + i, word.end(), back_inserter(new_word));
                break;
            }

            copy(word.begin() + i, wordIterator, back_inserter(new_word));
            i = distance(word.begin(), wordIterator);

            if (word[i] == first && i < word.size() -1 && word[i+1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = move(new_word);
        if (word.size() == 1) break;
        else {
            for (size_t i = 0; i < word.size()-1; ++i) {
                ranks[i] = m_bpe_ranks.find({word[i], word[i+1]});
            }
            ranks.resize(word.size()-1);
        }
    }

    return word;
}



vector<string> GPT2Tokenizer::tokenize(const string& text) {
    vector<string> result;
    for (auto match: ctre::range<pattern>(text)) {     // for token in re.findall(self.pat, text):    
        string token = match.to_string();         // match: ctre func.  token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        string byte_token;
        for (const auto& t: token) {
            byte_token += m_byte_encoder[t];
        }
        vector<string> bpe_result = bpe(byte_token);      // bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        result.reserve(result.size()+bpe_result.size());
        result.insert(result.end(), bpe_result.begin(), bpe_result.end());
    }

    return result;
}



vector<int64_t> GPT2Tokenizer::encode(const string& text) {
    vector<string> tokens = tokenize(text);
    vector<int64_t> token_ids;
    token_ids.reserve(tokens.size());
    transform(tokens.begin(), tokens.end(), back_inserter(token_ids),
                   [this](const string& token){
                       return m_encoder[token]; 
                   });
    return token_ids;
}

string GPT2Tokenizer::decode(const std::vector<int64_t>& token_ids) {
    std::string decoded_string;
    for (const auto& id: token_ids) {
        std::string decoded_token = m_decoder[id];
        for (size_t i = 0; i < decoded_token.size();) {
            int length = codepoint_length(decoded_token[i]);
            decoded_string += m_byte_decoder[decoded_token.substr(i, length)];
            i+=length;
        } 
    }
    return decoded_string;
}


size_t next_token_prediction(const std::unique_ptr<Ort::Session>& session_next, 
                             vector<int64_t>& token_ids, // cannot be const otherwise will confuse templated CreateTensor
                             const size_t vocab_size,
                             size_t seq_length,
                             int loop,
                             float temperature,
                             int k_value) {
    // Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);   

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // *************input*************:
    //  - input1: long tensor (?, batch_size, sequence_length)

    //*************************************************************************
    // ort_output: output1만 선택적 read, prediction_scores(=logits)
    vector<const char*> ort_input_name = {"input1"};
    vector<int64_t> ort_input_ids_shape {BATCH_SIZE, BATCH_SIZE,  static_cast<int64_t>(token_ids.size())};       // static_cast: 자료형변환. compile 타임에 형변환에 대한 타입 오류를 잡아줍니다.

    vector<const char*> ort_output_name = {"output1"};   // output1(prediction_scores)만 지정함.

    vector<Ort::Value> ort_input_tensor; 
    ort_input_tensor.push_back(
        Ort::Value::CreateTensor(memory_info, token_ids.data(), token_ids.size(), ort_input_ids_shape.data(), ort_input_ids_shape.size()));                    

    // Ort::Value ort_input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, token_ids.data(), token_ids.size(), ort_input_ids_shape.data(), ort_input_ids_shape.size());
   
    size_t input_count = 1;

    // ************* output *************
    //  - output1: float tensor (?, batch_size, sequence_length, vocab_size)
    vector<int64_t> prediction_scores_shape {BATCH_SIZE, BATCH_SIZE, static_cast<int64_t>(token_ids.size()), static_cast<int64_t>(vocab_size)};
    vector<float> prediction_scores(BATCH_SIZE * token_ids.size() * vocab_size);
    vector<Ort::Value> ort_output_tensor;
    ort_output_tensor.push_back(
        Ort::Value::CreateTensor(memory_info, prediction_scores.data(), prediction_scores.size(), prediction_scores_shape.data(), prediction_scores_shape.size()));

    size_t output_count = 1;    // logits값만 고려

    // std::vector<Ort::Value> ort_output_tensor = session_next.Run(Ort::RunOptions{nullptr}, ort_input_name.data(), &ort_input_tensor, input_count, ort_output_name.data(), output_count);
    // assert(ort_output_tensor.front().IsTensor());

    session_next->Run(   // C_api.h의 Run과 같다. (즉, GetApi의 ORT_API2_STATUS(Run, ~~ ) 과 같다.) // 사전 할당된 출력 목록이 있을 때 사용
        Ort::RunOptions{},
        ort_input_name.data(),
        ort_input_tensor.data(),
        // ort_input_tensor,    // 이용한 session->Run 인자에서는 받지 않는 형식.
        input_count,
        ort_output_name.data(),
        ort_output_tensor.data(),
        output_count
    );


    // temperature check source
    // cout<<"predict_scores : " <<prediction_scores.back()<< "\n";                                                                                                                                                                


    for (auto iter = prediction_scores.begin(); iter!= prediction_scores.end(); iter++) 
        *iter = *iter / temperature; // temperature 고려. 
    // auto prediction_scores = prediction_scores / temperature; wrong_code

    // cout<<"temp_predict_scores : " <<prediction_scores.back()<< "\n";                                                                                                                                                                


    // top_k_logits 함수 처리
    prediction_scores = topK(prediction_scores, k_value);
    // prediction_scores = topK(prediction_scores, 40);

    const auto new_word_prediction = view<float>{prediction_scores.begin() + (token_ids.size()-1) * vocab_size, 
                                                    prediction_scores.end()};                                                

    auto softmax = [](view<float> vec) {    
        std::transform(vec.begin(), vec.end(), vec.begin(), [](const float& el){ return std::exp(el); });   // e^el
        const float sum = std::accumulate(vec.begin(), vec.end(), 0.f); // accumulate third 인자: 'initial value of the sum', 즉 합의 초기값.
        std::transform(vec.begin(), vec.end(), vec.begin(), [sum](const float& el){ return el/sum; });
    };  

    softmax(new_word_prediction);   // softmax후의 결과로 transform 된다. 즉 log_probs로

    // check python source- sample: prev = torch.multinomial(num_samples=1)
    // const auto next_token_iter = *select_randomly(new_word_prediction.begin(), new_word_prediction.end()); 
    return std::distance(new_word_prediction.begin(), select_randomly(new_word_prediction.begin(), new_word_prediction.end()) ); // 위 next_token_iter의 distance를 return.
    
    // check python source- else: prev = torch.topk(k=1)
    // max_element는 if sample / else 구문에서의 top_k=1을 나타내는 알고리즘.
    // const auto next_token_iter = max_element(new_word_prediction.begin(), new_word_prediction.end()); 
    // return std::distance(new_word_prediction.begin(), next_token_iter); // 위 next_token_iter의 distance를 return.

    }

/* original Textgeneration py_source @ gpt2export_github

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

batch_size = 1
length = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Here is some text to encode : Hello World!"
tokens = np.array(tokenizer.encode(text))
context = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
prev = context
output = context

for i in range(length):
    outputs = model(prev)
    logits = outputs[0]
    logits = logits[:, -1, :]       // dim 설정만 바꿔주면 되는 코드
    log_probs = F.softmax(logits, dim=-1)
    _, prev = torch.topk(log_probs, k=1, dim=-1)
    output = torch.cat((output, prev), dim=1)

output = output[:, len(tokens):].tolist()
generated = 0
for i in range(batch_size):
    generated += 1
    text = tokenizer.decode(output[i])
    print(text)
*/

int main(int argc, char* argv[]) {


  static constexpr std::string_view vocab_file = "encoder.json";
  static constexpr std::string_view merges_file = "vocab.bpe";

  cxxopts::Options options("GPT2", "GPT2 implementation in C++ using Ort");

  options.add_options()
      ("t,text", "Initial text for GPT2", cxxopts::value<std::string>())
      ("n,number", "Number of new words to generate from initial text", cxxopts::value<size_t>()->default_value("1")) // default 값 1
      ("z,temperature", "Decide the temperature of GPT-2 model", cxxopts::value<float>()->default_value("1"))
      ("k,top_k", "Decide the k of top_k algorithm", cxxopts::value<int>()->default_value("40"))
      ("h,help", "Print usage")
  ;
  cxxopts::ParseResult result;
  
  try {
      result = options.parse(argc, argv);
  } catch (const cxxopts::OptionException& e) {
      std::cout << e.what() << "\n\n";
      std::cout << options.help() << std::endl;
      exit(0);
  }

  if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(0);
  }

  if (result.count("text") == 0) {
      std::cout << "Expected text input!\n\n";
      std::cout << options.help() << std::endl;
      exit(0);
  }
  const std::string text = result["text"].as<std::string>();
  const size_t generate = result["number"].as<size_t>();  
  const float temperature = result["temperature"].as<float>();  
  const float k_value = result["top_k"].as<int>();  
  // cxxopts -> option_parser end.

  
  auto tokenizer = unwrap(GPT2Tokenizer::load(vocab_file, merges_file), "Error initialising GPT2 tokenizer\n");

  auto vocab = tokenizer.m_encoder;


  //************************ for checking data load *************************
  cout<<"************************ for checking data load *************************"<< "\n";
  cout<<vocab.find("\u00a2")->first<<" "<<vocab.find("\u00a2")->second<< "\n";
  cout<<"!"<<" "<<vocab["!"]<< "\n\n";

  cout<<"unordered_map의 크기는 "<<vocab.size()<<" 입니다" << "\n\n";

  // string text = "Here is some text to encode Hello World";
  vector<string> text_result = tokenizer.tokenize(text); 

  auto token_ids = tokenizer.encode(text);

  cout<<"sentence : " <<text<< "\n\n";                                                                                                                                                                

  for (size_t i = 0; i < text_result.size(); i++)         
      cout << "initial_input_token[" << i << "]: " << text_result[i] << "\n";

  for (size_t i = 0; i < token_ids.size(); i++)          // input tensor check
      cout << "initial_input_tensor[" << i << "]: " << token_ids[i] << "\n";


  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
  // session (we also need to include cuda_provider_factory.h above which defines it)
  // #include "cuda_provider_factory.h"
  // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  //*************************************************************************
  // model from https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2

#ifdef _WIN32
  const wchar_t* model_path = L"gpt2-lm-head-1-dyn_out.onnx";
#else
  const char* model_path = "gpt2-lm-head-1-dyn_out.onnx";


#endif

  printf("\nUsing Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // initial input

  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);   // initial input
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());

    // onnx_model input_dim이 동적축일땐  오류로 잡는다.
    input_node_dims[0] = BATCH_SIZE;  //1 
    input_node_dims[1] = BATCH_SIZE;
    input_node_dims[2] = token_ids.size();

    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }

  //*************************************************************************
  // initial output
  // print model output layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator_output;

  // print number of model output nodes
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> output_node_names(num_output_nodes);
  std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of outputs = %zu\n", num_output_nodes);

  // iterate over all output nodes
  for (int i = 0; i < num_output_nodes; i++) {
    // print output node names
    char* output_name = session.GetOutputName(i, allocator_output);
    printf("Output %d : name=%s\n", i, output_name);
    output_node_names[i] = output_name;

    // print output node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Output %d : type=%d\n", i, type);

    // print output shapes/dims
    output_node_dims = tensor_info.GetShape();
    printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());

    // // when want to read real data
    // output_node_dims[1] = BATCH_SIZE;
    // output_node_dims[3] = token_ids.size();

    for (int j = 0; j < output_node_dims.size(); j++)
      printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
  }


  printf("\ninitial input tensor size %lu\n\n", token_ids.size());   //for check

//  Time checking for run the gpt-2 model
  clock_t start, finish;
  double duration;
  
  start = clock();

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, token_ids.data(), token_ids.size(), input_node_dims.data(), input_node_dims.size());
  assert(input_tensor.IsTensor());   



//*************initial_outputs************* 

/*  
    if input_text has 8 words(8 tokens), 

    outputs[1]: batch_size = 1, sequence_length = 8, vocab_size = 50257
    prediction_scores: Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). 
    It's a float tensor of size (batch_size, sequence_length, vocab_size). 

    outputs[2~13]: batch_size = 1, num_heads = 12, sequence_length = 8, hidden_size // num_attention_heads( = d_head: Dimensionality of the model’s heads) = 64
    past: pre-computed hidden-states. 
    It's a list of tensors (key and values in the attention blocks) of size (batch_size, num_heads, sequence_length, sequence_length), one per each layer. 

    size_t output1_tensor_size = 1 * 8 * 50267; // 402056
    size_t output2~13_tensor_size = 2 * 1 * 12 * 8 * 64; // 12288
    size_t output_tensor_size = (1 * 8 * 50267) + 12 * (2 * 1 * 12 * 8 * 64); // 549512
*/


  // score model & input tensor, get back output tensor
  std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 13);
  assert(output_tensors.size() == 13 && output_tensors.front().IsTensor());   

  finish = clock(); // time check end.
  
  duration = (double)(finish - start);
  cout << duration << "ms: gpt-2 model 1[time] running time" << endl;

  // Get pointer to output tensor float values
  
  size_t num_heads = 12;
  size_t seq_length = token_ids.size();
  size_t predict_score_data_num = BATCH_SIZE * seq_length * vocab.size();
  size_t present_score_data_num = 2 * BATCH_SIZE * num_heads * seq_length * 64;


// ************* for checking with pytorch result ***********************
  float* outputs[13];   
  for (int i = 0; i < 13; i++)
    // outputs[i] = output_tensors[i].front().GetTensorMutableData<float>(); // wrong code
    outputs[i] = output_tensors[i].GetTensorMutableData<float>();   // 위 seession.Run과 set로 사용가능한 것임.(seesin->Run과는 불가능. 다른 struct임.)


  // score the model, and print scores
  FILE *fp = fopen("Final_CXX_save_initial_output0.txt", "w");     
  FILE *fq = fopen("Final_CXX_save_initial_output1-12.txt", "w");


  // logits(=predict_scores) output[0]
  for (size_t i = 0; i < predict_score_data_num; i++){
    fprintf(fp, "logits(output[0]) [%lu] =  %f\n", i, outputs[0][i]);   
  }     
           
  // present_scores(output[1]~[11])
  for (int j = 0; j < 12; j++){    
      // printf("*** output[%lu] layer ***\n", j+1);
      // printf("\n");
      // printf("\n");

    for (size_t i = 0; i < present_score_data_num ; i++){
        // printf("output[%d] tensor [%lu] =  %f\n", j+2, i, outputs[j+1][i]);
        fprintf(fq, "present_score(output[%d]) [%lu] =  %f\n", j+1, i, outputs[j+1][i]);     
    }
                    
  }

  fclose(fp);   
  fclose(fq);


  printf("\n\nSave initial outputs --> Done!\n\n");


//   Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
  Ort::SessionOptions session_options_next{};
  session_options.SetIntraOpNumThreads(1);

  session_options_next.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//   Ort::Session session_next(env, model_path, session_options_next);
  auto session_next = std::make_unique<Ort::Session>(env, model_path, session_options_next);

  int loop = 0;
  
  //  Time checking for run the gpt-2 model
  clock_t start_loop, finish_loop;
  double duration_loop;
  
  start_loop = clock();

  for (size_t i = 0; i < generate; ++i) {  // generate: ("n,number", "Number of new words to generate from initial text", cxxopts::value<size_t>()->default_value("1"))
 
    token_ids.push_back(
        next_token_prediction(session_next, token_ids, tokenizer.vocab_size(), seq_length, loop, temperature, k_value));
        loop = loop + 1;
        cout << "token_ids_adding: " << token_ids.back() << '\n';  // for checking
  }    
   
  finish_loop = clock(); // time check end.
  
  duration_loop = (double)(finish_loop - start_loop) / CLOCKS_PER_SEC;
  cout << "\n\n" << duration_loop << "s: gpt-2 model_loop " << loop << " [times] running time" << endl;

  std::cout << "\nPrediction: \"" << tokenizer.decode(token_ids) << '\"' << '\n';


//   return 0;
}
