from transformers import BertForTokenClassification, DistilBertTokenizerFast, pipeline

# Carrega o modelo a partir de um diretório que só contém os arquivos a serem compartilhados
model = BertForTokenClassification.from_pretrained('neuralmind/bert-large-portuguese-cased')
tokenizer = DistilBertTokenizerFast.from_pretrained('neuralmind/bert-large-portuguese-cased'
                                                    , model_max_length=512
                                                    , do_lower_case=False
                                                    )

nlp = pipeline('ner', model=model, tokenizer=tokenizer)
nlp.model.config.id2label = {k: v.replace('L-', 'I-').replace('U-', 'B-') for k, v in nlp.model.config.id2label.items()}

result = nlp(
    """*FILHO DA MATA VOLTA LIMITE   *
_*30% PRA CÁ 70% PRA AI* _
:exclamation:️*SEM INTERMEDIÁRIO*:exclamation:️
*BANCO DO BRASIL CORRENTISTA* :yellow_heart:
*ITAU* :blue_heart:
*NUBANK*:purple_heart:
*BRADESCO* :heart:
*NEXT*  :heart:
*HIPERCARD* :heart:
*CREDCARD*:blue_heart:
*SANTANDER*:heart:
*TODOS ITAÚ PURO CRÉDITO LOJINHA*
CONSUTAR DEMAIS BANCO :bank:
REFERÊNCIAS
CONEXÃO DOS MONSTRO
OLD SCHOOL
KINGS DO 7
CÚPULA DOS MAGNATAS
DINASTIA
PLAY 7
CORRERIA SÓ OS CERTOS
E vários outros
*Trampo feito dentro do sistema e lembrando todo trampo tem seus risco , venha fazer a boa $*""".replace("\n",","),
    aggregation_strategy="MAX")
print(*result, sep="\n")
