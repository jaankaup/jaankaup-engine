struct McParams {
    base_position: vec4<f32>;
    isovalue: f32;
    cube_length: f32;
    for_future_usage1: f32;
    for_future_usage2: f32;
    noise_global_dimension: vec3<u32>; 
    noise_local_dimension: vec3<u32>; 
};

struct Vertex {
    v: vec4<f32>;
    n: vec4<f32>;
};

struct Cube {
    vertices: array<vec4<f32>, 8>;
    normals:  array<vec4<f32>, 8>;
};

@group(0)
@binding(0)
var<uniform> mc_uniform: McParams;

@group(0)
@binding(1)
var<storage, read_write> counter: array<atomic<u32>>; // atomic<32> doesn't work!

@group(0)
@binding(2)
var<storage, read> noise_values: array<f32>;

@group(0)
@binding(3)
var<storage, read_write> output: array<Vertex>;

var<private> cube: Cube;

//let edge_info: array<vec2<i32>, 12> = array<vec2<i32>, 12>(
// TODO: uniform!
var<private> edge_info: array<vec2<i32>, 12> = array<vec2<i32>, 12>(
        vec2<i32>(0,1), vec2<i32>(1,2), vec2<i32>(2,3), vec2<i32>(3,0), 
        vec2<i32>(4,5), vec2<i32>(5,6), vec2<i32>(6,7), vec2<i32>(7,4), 
        vec2<i32>(0,4), vec2<i32>(1,5), vec2<i32>(2,6), vec2<i32>(3,7)
); 

//let triTable: array<u32, 1280> = array<u32, 1280>(
// TODO: uniform!
var<private> triTable: array<u32, 1280> = array<u32, 1280>(
    16777215u , 16777215u , 16777215u , 16777215u , 16777215u ,
    2051u     , 16777215u , 16777215u , 16777215u , 16777215u ,
    265u      , 16777215u , 16777215u , 16777215u , 16777215u ,
    67587u ,591873u ,16777215u ,16777215u ,16777215u ,
    66058u ,16777215u ,16777215u ,16777215u ,16777215u ,
    2051u ,66058u ,16777215u ,16777215u ,16777215u ,
    590346u ,521u ,16777215u ,16777215u ,16777215u ,
    133123u ,133640u ,657672u ,16777215u ,16777215u ,
    199426u ,16777215u ,16777215u ,16777215u ,16777215u ,
    2818u ,527104u ,16777215u ,16777215u ,16777215u ,
    67840u ,131851u ,16777215u ,16777215u ,16777215u ,
    68354u ,67851u ,591883u ,16777215u ,16777215u ,
    199169u ,723459u ,16777215u ,16777215u ,16777215u ,
    2561u ,2058u ,527114u ,16777215u ,16777215u ,
    198912u ,199433u ,723465u ,16777215u ,16777215u ,
    591882u ,657419u ,16777215u ,16777215u ,16777215u ,
    263944u ,16777215u ,16777215u ,16777215u ,16777215u ,
    262912u ,459524u ,16777215u ,16777215u ,16777215u ,
    265u ,525319u ,16777215u ,16777215u ,16777215u ,
    262409u ,263937u ,459521u ,16777215u ,16777215u ,
    66058u ,525319u ,16777215u ,16777215u ,16777215u ,
    197639u ,196612u ,66058u ,16777215u ,16777215u ,
    590346u ,589826u ,525319u ,16777215u ,16777215u ,
    133641u ,133383u ,132867u ,461060u ,16777215u ,
    525319u ,199426u ,16777215u ,16777215u ,16777215u ,
    721927u ,721412u ,131076u ,16777215u ,16777215u ,
    589825u ,525319u ,131851u ,16777215u ,16777215u ,
    263947u ,590859u ,592642u ,590337u ,16777215u ,
    199169u ,199434u ,460804u ,16777215u ,16777215u ,
    68362u ,66571u ,65540u ,461572u ,16777215u ,
    263944u ,589835u ,592650u ,720899u ,16777215u ,
    263947u ,264969u ,592650u ,16777215u ,16777215u ,
    591108u ,16777215u ,16777215u ,16777215u ,16777215u ,
    591108u ,2051u ,16777215u ,16777215u ,16777215u ,
    1284u ,66816u ,16777215u ,16777215u ,16777215u ,
    525572u ,525061u ,196869u ,16777215u ,16777215u ,
    66058u ,591108u ,16777215u ,16777215u ,16777215u ,
    196616u ,66058u ,264453u ,16777215u ,16777215u ,
    328202u ,328706u ,262146u ,16777215u ,16777215u ,
    133637u ,197125u ,197892u ,197640u ,16777215u ,
    591108u ,131851u ,16777215u ,16777215u ,16777215u ,
    2818u ,2059u ,264453u ,16777215u ,16777215u ,
    1284u ,261u ,131851u ,16777215u ,16777215u ,
    131333u ,132360u ,133131u ,264197u ,16777215u ,
    656139u ,655619u ,591108u ,16777215u ,16777215u ,
    264453u ,2049u ,526849u ,527114u ,16777215u ,
    328704u ,327691u ,330506u ,720899u ,16777215u ,
    328712u ,329738u ,657419u ,16777215u ,16777215u ,
    591624u ,329481u ,16777215u ,16777215u ,16777215u ,
    590592u ,591107u ,329475u ,16777215u ,16777215u ,
    1800u ,263u ,66823u ,16777215u ,16777215u ,
    66819u ,197895u ,16777215u ,16777215u ,16777215u ,
    591624u ,591111u ,655618u ,16777215u ,16777215u ,
    655618u ,591104u ,328448u ,329475u ,16777215u ,
    524290u ,524805u ,525575u ,656642u ,16777215u ,
    133637u ,132355u ,197895u ,16777215u ,16777215u ,
    461061u ,460809u ,199426u ,16777215u ,16777215u ,
    591111u ,591618u ,590336u ,132875u ,16777215u ,
    131851u ,264u ,67336u ,66823u ,16777215u ,
    721409u ,721159u ,459013u ,16777215u ,16777215u ,
    591112u ,525575u ,655619u ,656139u ,16777215u ,
    329472u ,327689u ,461568u ,65546u ,723456u ,
    723456u ,720899u ,656640u ,524295u ,329472u ,
    723461u ,461573u ,16777215u ,16777215u ,16777215u ,
    656901u ,16777215u ,16777215u ,16777215u ,16777215u ,
    2051u ,330246u ,16777215u ,16777215u ,16777215u ,
    589825u ,330246u ,16777215u ,16777215u ,16777215u ,
    67587u ,67848u ,330246u ,16777215u ,16777215u ,
    67077u ,132609u ,16777215u ,16777215u ,16777215u ,
    67077u ,66054u ,196616u ,16777215u ,16777215u ,
    591365u ,589830u ,518u ,16777215u ,16777215u ,
    329992u ,329730u ,328198u ,197128u ,16777215u ,
    131851u ,656901u ,16777215u ,16777215u ,16777215u ,
    720904u ,721408u ,656901u ,16777215u ,16777215u ,
    265u ,131851u ,330246u ,16777215u ,16777215u ,
    330246u ,67842u ,592642u ,591883u ,16777215u ,
    393995u ,394499u ,327939u ,16777215u ,16777215u ,
    2059u ,2821u ,1281u ,330502u ,16777215u ,
    199430u ,774u ,1541u ,1289u ,16777215u ,
    394505u ,395531u ,723208u ,16777215u ,16777215u ,
    330246u ,263944u ,16777215u ,16777215u ,16777215u ,
    262912u ,263939u ,394506u ,16777215u ,16777215u ,
    67840u ,330246u ,525319u ,16777215u ,16777215u ,
    656901u ,67847u ,67331u ,461060u ,16777215u ,
    393474u ,394497u ,263944u ,16777215u ,16777215u ,
    66053u ,328198u ,196612u ,197639u ,16777215u ,
    525319u ,589829u ,1541u ,518u ,16777215u ,
    459529u ,461060u ,197129u ,329990u ,132617u ,
    199426u ,460804u ,656901u ,16777215u ,16777215u ,
    330246u ,263938u ,262656u ,132875u ,16777215u ,
    265u ,263944u ,131851u ,330246u ,16777215u ,
    590337u ,592642u ,590859u ,461572u ,330246u ,
    525319u ,199429u ,197889u ,330502u ,16777215u ,
    327947u ,330502u ,65547u ,461572u ,1035u ,
    1289u ,1541u ,774u ,722435u ,525319u ,
    394505u ,395531u ,263945u ,461577u ,16777215u ,
    656393u ,394250u ,16777215u ,16777215u ,16777215u ,
    264710u ,264458u ,2051u ,16777215u ,16777215u ,
    655361u ,656896u ,394240u ,16777215u ,16777215u ,
    525057u ,524550u ,525828u ,393482u ,16777215u ,
    66569u ,66052u ,132612u ,16777215u ,16777215u ,
    196616u ,66057u ,132105u ,132612u ,16777215u ,
    516u ,262662u ,16777215u ,16777215u ,16777215u ,
    525058u ,524804u ,262662u ,16777215u ,16777215u ,
    656393u ,656900u ,721411u ,16777215u ,16777215u ,
    2050u ,133131u ,264458u ,264710u ,16777215u ,
    199426u ,262u ,1540u ,393482u ,16777215u ,
    394241u ,393482u ,264193u ,131339u ,527105u ,
    591364u ,590598u ,590083u ,722435u ,16777215u ,
    527105u ,524544u ,722433u ,590084u ,394241u ,
    199430u ,198144u ,1540u ,16777215u ,16777215u ,
    394248u ,722440u ,16777215u ,16777215u ,16777215u ,
    461318u ,460810u ,526602u ,16777215u ,16777215u ,
    1795u ,2567u ,2314u ,395018u ,16777215u ,
    656903u ,68103u ,67336u ,67584u ,16777215u ,
    656903u ,657153u ,67331u ,16777215u ,16777215u ,
    66054u ,67080u ,67593u ,525831u ,16777215u ,
    132617u ,133377u ,395017u ,2307u ,459529u ,
    460800u ,458758u ,393218u ,16777215u ,16777215u ,
    459522u ,395010u ,16777215u ,16777215u ,16777215u ,
    131851u ,656904u ,657417u ,525831u ,16777215u ,
    131079u ,132875u ,2311u ,395018u ,592391u ,
    67584u ,67336u ,68103u ,395018u ,131851u ,
    721409u ,721159u ,656897u ,395009u ,16777215u ,
    526598u ,525831u ,590086u ,722435u ,66310u ,
    2305u ,722439u ,16777215u ,16777215u ,16777215u ,
    460800u ,458758u ,199424u ,722432u ,16777215u ,
    461574u ,16777215u ,16777215u ,16777215u ,16777215u ,
    460299u ,16777215u ,16777215u ,16777215u ,16777215u ,
    196616u ,722694u ,16777215u ,16777215u ,16777215u ,
    265u ,722694u ,16777215u ,16777215u ,16777215u ,
    524553u ,525057u ,722694u ,16777215u ,16777215u ,
    655618u ,396039u ,16777215u ,16777215u ,16777215u ,
    66058u ,196616u ,396039u ,16777215u ,16777215u ,
    133376u ,133641u ,396039u ,16777215u ,16777215u ,
    396039u ,133635u ,657411u ,657672u ,16777215u ,
    459267u ,393735u ,16777215u ,16777215u ,16777215u ,
    458760u ,460288u ,393728u ,16777215u ,16777215u ,
    132870u ,131847u ,265u ,16777215u ,16777215u ,
    67074u ,67590u ,67848u ,526086u ,16777215u ,
    657158u ,655623u ,66311u ,16777215u ,16777215u ,
    657158u ,67338u ,67591u ,65544u ,16777215u ,
    775u ,1802u ,2569u ,395783u ,16777215u ,
    460298u ,461320u ,526857u ,16777215u ,16777215u ,
    395268u ,722950u ,16777215u ,16777215u ,16777215u ,
    198155u ,196614u ,1030u ,16777215u ,16777215u ,
    525835u ,525318u ,589825u ,16777215u ,16777215u ,
    590854u ,591363u ,590593u ,721670u ,16777215u ,
    395268u ,396040u ,133633u ,16777215u ,16777215u ,
    66058u ,196619u ,1547u ,1030u ,16777215u ,
    264968u ,263691u ,521u ,133641u ,16777215u ,
    657667u ,656130u ,590851u ,721670u ,263683u ,
    524803u ,525314u ,263682u ,16777215u ,16777215u ,
    1026u ,263682u ,16777215u ,16777215u ,16777215u ,
    67840u ,131844u ,132102u ,262920u ,16777215u ,
    67844u ,66562u ,132102u ,16777215u ,16777215u ,
    524547u ,525825u ,525318u ,395777u ,16777215u ,
    655616u ,655366u ,393220u ,16777215u ,16777215u ,
    263683u ,262920u ,395779u ,777u ,657667u ,
    657668u ,395780u ,16777215u ,16777215u ,16777215u ,
    264453u ,460299u ,16777215u ,16777215u ,16777215u ,
    2051u ,264453u ,722694u ,16777215u ,16777215u ,
    327681u ,328704u ,460299u ,16777215u ,16777215u ,
    722694u ,525060u ,197892u ,196869u ,16777215u ,
    591108u ,655618u ,460299u ,16777215u ,16777215u ,
    396039u ,66058u ,2051u ,264453u ,16777215u ,
    460299u ,328714u ,262666u ,262146u ,16777215u ,
    197640u ,197892u ,197125u ,656642u ,722694u ,
    459267u ,460290u ,328713u ,16777215u ,16777215u ,
    591108u ,2054u ,1538u ,395271u ,16777215u ,
    198146u ,198406u ,66816u ,328704u ,16777215u ,
    393736u ,395271u ,131336u ,264197u ,66824u ,
    591108u ,655622u ,67334u ,66311u ,16777215u ,
    67082u ,67334u ,65543u ,526080u ,591108u ,
    262154u ,264709u ,778u ,395783u ,198410u ,
    460298u ,461320u ,328714u ,264202u ,16777215u ,
    395525u ,396041u ,722953u ,16777215u ,16777215u ,
    198155u ,1539u ,1286u ,2309u ,16777215u ,
    2824u ,1291u ,261u ,329227u ,16777215u ,
    396035u ,393989u ,328449u ,16777215u ,16777215u ,
    66058u ,591115u ,592648u ,722182u ,16777215u ,
    2819u ,1547u ,2310u ,329225u ,66058u ,
    722949u ,722182u ,524293u ,656642u ,517u ,
    396035u ,393989u ,133635u ,656643u ,16777215u ,
    329737u ,328200u ,329218u ,198658u ,16777215u ,
    591110u ,591360u ,1538u ,16777215u ,16777215u ,
    66824u ,67584u ,329224u ,198658u ,393736u ,
    66822u ,131334u ,16777215u ,16777215u ,16777215u ,
    66310u ,67082u ,198662u ,329225u ,526598u ,
    655616u ,655366u ,591104u ,329216u ,16777215u ,
    776u ,329226u ,16777215u ,16777215u ,16777215u ,
    656646u ,16777215u ,16777215u ,16777215u ,16777215u ,
    722186u ,460043u ,16777215u ,16777215u ,16777215u ,
    722186u ,722693u ,525056u ,16777215u ,16777215u ,
    330503u ,330251u ,67840u ,16777215u ,16777215u ,
    657157u ,658183u ,591873u ,525057u ,16777215u ,
    721154u ,722689u ,460033u ,16777215u ,16777215u ,
    2051u ,66055u ,67333u ,459275u ,16777215u ,
    591621u ,590343u ,589826u ,133895u ,16777215u ,
    460034u ,459275u ,329986u ,197128u ,591874u ,
    132362u ,131845u ,198405u ,16777215u ,16777215u ,
    524800u ,525570u ,526085u ,655877u ,16777215u ,
    589825u ,330243u ,328455u ,199170u ,16777215u ,
    591874u ,590337u ,526082u ,655877u ,460034u ,
    66309u ,198405u ,16777215u ,16777215u ,16777215u ,
    2055u ,1793u ,67333u ,16777215u ,16777215u ,
    589827u ,590597u ,328455u ,16777215u ,16777215u ,
    591879u ,329991u ,16777215u ,16777215u ,16777215u ,
    329732u ,330248u ,658184u ,16777215u ,16777215u ,
    327684u ,330496u ,330251u ,721664u ,16777215u ,
    265u ,525322u ,526859u ,656389u ,16777215u ,
    658180u ,656389u ,721668u ,590849u ,196868u ,
    132353u ,133125u ,133896u ,263432u ,16777215u ,
    1035u ,2819u ,263435u ,133889u ,327947u ,
    517u ,1289u ,133893u ,263432u ,722949u ,
    590853u ,133891u ,16777215u ,16777215u ,16777215u ,
    132362u ,197890u ,197637u ,198660u ,16777215u ,
    330242u ,328196u ,262656u ,16777215u ,16777215u ,
    199170u ,197898u ,198661u ,263432u ,265u ,
    330242u ,328196u ,67842u ,590850u ,16777215u ,
    525317u ,525571u ,197889u ,16777215u ,16777215u ,
    1029u ,65541u ,16777215u ,16777215u ,16777215u ,
    525317u ,525571u ,589829u ,773u ,16777215u ,
    590853u ,16777215u ,16777215u ,16777215u ,16777215u ,
    264967u ,264459u ,592395u ,16777215u ,16777215u ,
    2051u ,264455u ,592647u ,592395u ,16777215u ,
    68107u ,68356u ,66560u ,459787u ,16777215u ,
    196868u ,197640u ,68100u ,459787u ,658180u ,
    264967u ,592644u ,590347u ,590082u ,16777215u ,
    591620u ,592647u ,590091u ,133889u ,2051u ,
    722692u ,721922u ,132096u ,16777215u ,16777215u ,
    722692u ,721922u ,525060u ,197124u ,16777215u ,
    133386u ,132873u ,131847u ,459785u ,16777215u ,
    592391u ,591620u ,655879u ,526080u ,131079u ,
    198410u ,199170u ,459786u ,68096u ,262154u ,
    68098u ,526084u ,16777215u ,16777215u ,16777215u ,
    264449u ,262407u ,459011u ,16777215u ,16777215u ,
    264449u ,262407u ,2049u ,526081u ,16777215u ,
    262147u ,459779u ,16777215u ,16777215u ,16777215u ,
    264199u ,16777215u ,16777215u ,16777215u ,16777215u ,
    592392u ,658184u ,16777215u ,16777215u ,16777215u ,
    196617u ,198923u ,723210u ,16777215u ,16777215u ,
    266u ,2568u ,526859u ,16777215u ,16777215u ,
    196874u ,721674u ,16777215u ,16777215u ,16777215u ,
    66059u ,68361u ,592648u ,16777215u ,16777215u ,
    196617u ,198923u ,66057u ,133897u ,16777215u ,
    523u ,524299u ,16777215u ,16777215u ,16777215u ,
    197131u ,16777215u ,16777215u ,16777215u ,16777215u ,
    131848u ,133130u ,657417u ,16777215u ,16777215u ,
    592386u ,2306u ,16777215u ,16777215u ,16777215u ,
    131848u ,133130u ,264u ,68104u ,16777215u ,
    68098u ,16777215u ,16777215u ,16777215u ,16777215u ,
    66312u ,590088u ,16777215u ,16777215u ,16777215u ,
    2305u ,16777215u ,16777215u ,16777215u ,16777215u ,
    776u ,16777215u ,16777215u ,16777215u ,16777215u ,
    16777215u ,16777215u ,16777215u ,16777215u ,16777215u
);

fn encode3Dmorton32(x: u32, y: u32, z: u32) -> u32 {
    var x_temp = (x      | (x      << 16u)) & 0x030000FFu;
        x_temp = (x_temp | (x_temp <<  8u)) & 0x0300F00Fu;
        x_temp = (x_temp | (x_temp <<  4u)) & 0x030C30C3u;
        x_temp = (x_temp | (x_temp <<  2u)) & 0x09249249u;

    var y_temp = (y      | (y      << 16u)) & 0x030000FFu;
        y_temp = (y_temp | (y_temp <<  8u)) & 0x0300F00Fu;
        y_temp = (y_temp | (y_temp <<  4u)) & 0x030C30C3u;
        y_temp = (y_temp | (y_temp <<  2u)) & 0x09249249u;

    var z_temp = (z      | (z      << 16u)) & 0x030000FFu;
        z_temp = (z_temp | (z_temp <<  8u)) & 0x0300F00Fu;
        z_temp = (z_temp | (z_temp <<  4u)) & 0x030C30C3u;
        z_temp = (z_temp | (z_temp <<  2u)) & 0x09249249u;

    return x_temp | (y_temp << 1u) | (z_temp << 2u);
}

fn get_third_bits32(m: u32) -> u32 {
    var x = m & 0x9249249u;
    x = (x ^ (x >> 2u))  & 0x30c30c3u;
    x = (x ^ (x >> 4u))  & 0x300f00fu;
    x = (x ^ (x >> 8u))  & 0x30000ffu;
    x = (x ^ (x >> 16u)) & 0x3ffu;

    return x;
}

fn decode3Dmorton32(m: u32) -> vec3<u32> {
    return vec3<u32>(
        get_third_bits32(m),
        get_third_bits32(m >> 1u),
        get_third_bits32(m >> 2u)
   );
}


// Noise functions copied from https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83 and converted to wgsl.

fn hash(n: f32) -> f32 {
    return fract(sin(n) * 10000.0);
}

fn hash_v2(p: vec2<f32>) -> f32 {
    return fract(10000.0 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x))));
}

fn noise(x: f32) -> f32 {
    let i: f32 = floor(x);
    let f: f32 = fract(x);
    let u: f32 = f * f * (3.0 - 2.0 * f);
    return mix(hash(i), hash(i + 1.0), u);
}

fn noise2(x: vec2<f32>) -> f32 {

	let i: vec2<f32> = floor(x);
	let f: vec2<f32> = fract(x);

	// Four corners in 2D of a tile
	let a: f32 = hash_v2(i);
	let b: f32 = hash_v2(i + vec2<f32>(1.0, 0.0));
	let c: f32 = hash_v2(i + vec2<f32>(0.0, 1.0));
	let d: f32 = hash_v2(i + vec2<f32>(1.0, 1.0));

	let u: vec2<f32> = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

fn noise3(x: vec3<f32>) -> f32 {

	let st = vec3<f32>(110.0, 241.0, 171.0);

	let i = floor(x);
	let f = fract(x);

    	let n = dot(i, st);


	let u = f * f * (3.0 - 2.0 * f);
	return mix(mix(mix( hash(n + dot(st, vec3<f32>(0.0, 0.0, 0.0))), hash(n + dot(st, vec3<f32>(1.0, 0.0, 0.0))), u.x),
                   mix( hash(n + dot(st, vec3<f32>(0.0, 1.0, 0.0))), hash(n + dot(st, vec3<f32>(1.0, 1.0, 0.0))), u.x), u.y),
               mix(mix( hash(n + dot(st, vec3<f32>(0.0, 0.0, 1.0))), hash(n + dot(st, vec3<f32>(1.0, 0.0, 1.0))), u.x),
                   mix( hash(n + dot(st, vec3<f32>(0.0, 1.0, 1.0))), hash(n + dot(st, vec3<f32>(1.0, 1.0, 1.0))), u.x), u.y), u.z);
}

let NUM_OCTAVES: u32 = 5u;

fn fbm(x: f32) -> f32 {

    var v: f32 = 0.0;
    var a: f32 = 0.5;
    var xx: f32 = x; 
    let shift: f32 = 100.0;
    for (var i: u32 = 0u; i < NUM_OCTAVES; i = i + 1u) {
    	v = a + a * noise(xx);
    	xx = xx * 2.0 + shift;
    	a = a * 0.5;
    }
    return v;
}


fn fbm2(x: vec2<f32>) -> f32 {

    let shift = vec2<f32>(100.0);
    let rot = mat2x2<f32>(vec2<f32>(cos(0.5), sin(0.5)), vec2<f32>(-sin(0.5), cos(0.50)));
    
    var v: f32 = 0.0;
    var a: f32 = 0.5;
    var xx: vec2<f32> = x; 
    
    for (var i: u32 = 0u; i < NUM_OCTAVES; i = i + 1u) {
        v = v + a * noise2(xx);
        xx = rot * xx * 2.0 + shift;
        a = a * 0.5;
    }
    return v;
}

fn fbm3(x: vec3<f32>) -> f32 {

    let shift: f32 = 100.0;

    var v: f32 = 0.0;
    var a: f32 = 0.5;
    var xx: vec3<f32> = x; 

    for (var i: u32 = 0u; i < NUM_OCTAVES; i = i + 1u) {
    	v = a + a * noise3(xx);
    	xx = xx * 2.0 + shift;
    	a = a * 0.5;
    }
    return v;
}

// Marching cubes.


fn calculate_density(v: vec3<i32>) -> f32 {

    if (v.x < 0 ||
        v.y < 0 ||
        v.z < 0 ||
        v.x >= i32(mc_uniform.noise_local_dimension.x) ||
        v.y >= i32(mc_uniform.noise_local_dimension.y) ||
        v.z >= i32(mc_uniform.noise_local_dimension.z)) { return 0.0; }

    // if (encode3Dmorton32(u32(v.x), u32(v.y), u32(v.z)) > 64u*64u*64u) { return 0.0; }

    return noise_values[
	encode3Dmorton32(
            u32(v.x), u32(v.y), u32(v.z)
        )
    ];
}

fn calculate_case() -> u32 {

  var result: u32 = 0u;

  result = result | (select(0u, 1u,  cube.vertices[7].a < mc_uniform.isovalue) << 7u);
  result = result | (select(0u, 1u,  cube.vertices[6].a < mc_uniform.isovalue) << 6u);
  result = result | (select(0u, 1u,  cube.vertices[5].a < mc_uniform.isovalue) << 5u);
  result = result | (select(0u, 1u,  cube.vertices[4].a < mc_uniform.isovalue) << 4u);
  result = result | (select(0u, 1u,  cube.vertices[3].a < mc_uniform.isovalue) << 3u);
  result = result | (select(0u, 1u,  cube.vertices[2].a < mc_uniform.isovalue) << 2u);
  result = result | (select(0u, 1u,  cube.vertices[1].a < mc_uniform.isovalue) << 1u);
  result = result | (select(0u, 1u,  cube.vertices[0].a < mc_uniform.isovalue) << 0u);

  return result;
}

fn calculate_normal(pos: vec3<i32>) -> vec3<f32> {

  // let r = pos + vec3<f32>(1, 0, 0);
  // let l = pos - vec3<f32>(1, 0, 0);
  // let u = pos + vec3<f32>(0, 1, 0);
  // let d = pos - vec3<f32>(0, 1, 0);
  // let z_m = pos + vec3<f32>(0, 0, 1);
  // let z_p = pos - vec3<f32>(0, 0, 1);

  // var right: f32   = 0.0;
  // var left: f32    = 0.0;
  // var up: f32      = 0.0;
  // var down: f32    = 0.0;
  // var z_minus: f32 = 0.0;
  // var z: f32       = 0.0;

  // if (r < 0) 

  let right: f32   = calculate_density(pos + vec3<i32>(1, 0, 0));
  let left: f32    = calculate_density(pos - vec3<i32>(1, 0, 0));
  let up: f32      = calculate_density(pos + vec3<i32>(0, 1, 0));
  let down: f32    = calculate_density(pos - vec3<i32>(0, 1, 0));
  let z_minus: f32 = calculate_density(pos + vec3<i32>(0, 0, 1));
  let z: f32       = calculate_density(pos - vec3<i32>(0, 0, 1));

  var grad: vec3<f32>;
  grad.x = right - left;
  grad.y = up - down;
  grad.z = z_minus - z;
  //grad.z = z - z_minus;
  return normalize(grad); // TODO: check if is this necessery
}

fn interpolateV(va: vec4<f32>, vb: vec4<f32>) -> vec4<f32> {

    // TODO: eliminate conditionals branches.

    if (abs(mc_uniform.isovalue - va.w) < 0.0001) {
       return vec4<f32>(va.xyz, 1.0);
    }
    else if (abs(mc_uniform.isovalue - vb.w) < 0.00001) {
       return vec4<f32>(vb.xyz, 1.0);
    }
    else if (abs(va.w-vb.w) < 0.00001) {
       return vec4<f32>(va.xyz, 1.0);
    }
    
    else {
      var p: vec4<f32>;
      var mu: f32 = (mc_uniform.isovalue - va.w) / (vb.w - va.w);
      p.x = va.x + mu * (vb.x - va.x);
      p.y = va.y + mu * (vb.y - va.y);
      p.z = va.z + mu * (vb.z - va.z);
      p.w = 1.0;
      return p;
    }
}

fn interpolateN(na: vec4<f32>, nb: vec4<f32>, densityA: f32, densityB: f32) -> vec4<f32> {

    // TODO: eliminate conditionals.

    if (abs(mc_uniform.isovalue - densityA) < 0.00001) {
        return vec4<f32>(normalize(na.xyz), 0.0);
    }
    else if (abs(mc_uniform.isovalue - densityB) < 0.00001) {
        return vec4<f32>(normalize(nb.xyz), 0.0);
     }
    else if (abs(densityA-densityB) < 0.00001) {
        return vec4<f32>(normalize(na.xyz), 0.0);
     }
     
    else {
      let mu: f32 = (mc_uniform.isovalue - densityA) / (densityB - densityA);
      let x: f32 = na.x + mu * (nb.x - na.x);
      let y: f32 = na.y + mu * (nb.y - na.y);
      let z: f32 = na.z + mu * (nb.z - na.z);
      return vec4<f32>(normalize(vec3<f32>(x, y, z)), 0.0);
    }
}

fn createVertex(edgeValue: i32, arrayIndex: i32) {

    let edge = edge_info[edgeValue];
 
    let vert_a: vec4<f32> = cube.vertices[edge.x];
    let vert_b: vec4<f32> = cube.vertices[edge.y];

    var v: Vertex;

    v.v = interpolateV(vert_a, vert_b);
    v.n = interpolateN(cube.normals[edge.x], cube.normals[edge.y], vert_a.a, vert_b.a);

    output[arrayIndex] = v;
}

fn index1D_to_index3D(global_index: vec3<u32>, x_dim: u32, y_dim: u32) -> vec3<u32> {
	var index: u32 = global_index.x;
	var wh: u32 = x_dim * y_dim;
	let z: u32 = index / wh;
	index = index - z * wh;
	let y: u32 = index / x_dim;
	index = index - y * x_dim;
	let x: u32 = index;
	return vec3<u32>(x, y, z);	
}

@stage(compute)
@workgroup_size(256,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32) {

    // Create and scale cube base position.
    // let coordinate3D = vec3<i32>(decode3Dmorton32(global_id.x + 65536u * global_id.y));
    let coordinate3D = vec3<i32>(decode3Dmorton32(global_id.x));

    //let position = vec3<f32>(f32(global_id.x), f32(global_id.y), f32(global_id.z)) * mc_uniform.cube_length + mc_uniform.base_position.xyz;
    let position = coordinate3D; //global_id.x + 65536 * global_id.y;

    // Create cube corner coordinates. 
    // let p0 = position;
    // let p3 = position + vec3<f32>(0.0                      , mc_uniform.cube_length   , 0.0);
    // let p2 = position + vec3<f32>(mc_uniform.cube_length   , mc_uniform.cube_length   , 0.0);
    // let p1 = position + vec3<f32>(mc_uniform.cube_length   , 0.0                      , 0.0);
    // let p4 = position + vec3<f32>(0.0                      , 0.0                      , mc_uniform.cube_length);
    // let p7 = position + vec3<f32>(0.0                      , mc_uniform.cube_length   , mc_uniform.cube_length);
    // let p6 = position + vec3<f32>(mc_uniform.cube_length   , mc_uniform.cube_length   , mc_uniform.cube_length);
    // let p5 = position + vec3<f32>(mc_uniform.cube_length   , 0.0                      , mc_uniform.cube_length);

    let p0 = position;
    let p3 = position + vec3<i32>(0 , 1 , 0);
    let p2 = position + vec3<i32>(1 , 1 , 0);
    let p1 = position + vec3<i32>(1 , 0 , 0);
    let p4 = position + vec3<i32>(0 , 0 , 1);
    let p7 = position + vec3<i32>(0 , 1 , 1);
    let p6 = position + vec3<i32>(1 , 1 , 1);
    let p5 = position + vec3<i32>(1 , 0 , 1);

    // Cube corner positions and density values.
    cube.vertices[0] = vec4<f32>(vec3<f32>(p0) * mc_uniform.cube_length + mc_uniform.base_position.xyz, calculate_density(p0));
    cube.vertices[1] = vec4<f32>(vec3<f32>(p1) * mc_uniform.cube_length + mc_uniform.base_position.xyz, calculate_density(p1));
    cube.vertices[2] = vec4<f32>(vec3<f32>(p2) * mc_uniform.cube_length + mc_uniform.base_position.xyz, calculate_density(p2));
    cube.vertices[3] = vec4<f32>(vec3<f32>(p3) * mc_uniform.cube_length + mc_uniform.base_position.xyz, calculate_density(p3));
    cube.vertices[4] = vec4<f32>(vec3<f32>(p4) * mc_uniform.cube_length + mc_uniform.base_position.xyz, calculate_density(p4));
    cube.vertices[5] = vec4<f32>(vec3<f32>(p5) * mc_uniform.cube_length + mc_uniform.base_position.xyz, calculate_density(p5));
    cube.vertices[6] = vec4<f32>(vec3<f32>(p6) * mc_uniform.cube_length + mc_uniform.base_position.xyz, calculate_density(p6));
    cube.vertices[7] = vec4<f32>(vec3<f32>(p7) * mc_uniform.cube_length + mc_uniform.base_position.xyz, calculate_density(p7));
    
    // Calculate the cube case number.
    let cube_case = calculate_case();
      
    // This cube doesn't create any triangles.
    if (cube_case == 0u || cube_case == 255u) { return; }

    // Calculate normals for cube corners.
    cube.normals[0] = vec4<f32>(calculate_normal(p0), 0.0);
    cube.normals[1] = vec4<f32>(calculate_normal(p1), 0.0);
    cube.normals[2] = vec4<f32>(calculate_normal(p2), 0.0);
    cube.normals[3] = vec4<f32>(calculate_normal(p3), 0.0);
    cube.normals[4] = vec4<f32>(calculate_normal(p4), 0.0);
    cube.normals[5] = vec4<f32>(calculate_normal(p5), 0.0);
    cube.normals[6] = vec4<f32>(calculate_normal(p6), 0.0);
    cube.normals[7] = vec4<f32>(calculate_normal(p7), 0.0);

    let OFFSET: u32 = 5u;

    var i: u32 = 0u;

    loop {
 	if (i == 5u) { break; }

        let base_index: u32 = triTable[cube_case * OFFSET + i];

        if (base_index != 16777215u) { 

            let index = atomicAdd(&counter[0], 3u);

            // Create the triangle vertices and normals.
            createVertex(i32((base_index & 0xff0000u) >> 16u), i32(index));
            createVertex(i32((base_index & 0xff00u) >> 8u)   , i32(index+1u));
            createVertex(i32( base_index & 0xffu),            i32(index+2u));
        }
	i = i + 1u;
    }
}
