# Solidus Blind Test

Playground for Solidus Blind Test.

## Data Set  

There are 3 files in the data set:

1. 	`Dataset Bitmart.csv` - Labeled.  
	To be used for model training.
2.	`Dataset Solidus.csv` - Un Labeled.  
	To be used for inference.
1. 	`Dataset Wormhole.csv` - Labeled.  
	To be used for model training.


## CSV Format

This is verions `0.1` of the CSV / Data Frame format. See [`TemplateCsv.csv`](https://github.com/CyVers-AI/BlockChainAttacksDataSet/blob/main/TemplateCsv.csv).

 -	RAW Data Columns
	-	`Transaction ID` - The _Hash_ of the transaction or any one to one identifier.
	-	`Transaction Time` - The time of the transaction. The format is `YYYY_MM_DD_HH_MM_SS`.
	-	`Block Time` - The time of the block. The format is `YYYY_MM_DD_HH_MM_SS`.
	-	`Sender ID` - A one to one identifier of the sender. Usually the public key.
	-	`Receiver ID` - A one to one identifier of the receiver. Usually the public key.
	-	`Receiver Type` - A categorical value of the type of receiver (For example: `WALLET`, `CONTRACT`, etc...).
	-	`Amount` - The amount of the currency transfer. In the native currency (_Token_).  
	-	`Currency` - The token (Currency) used by transaction. It will be a categorical variable (For instance `ETH`, `BITCOIN`, etc...).
	-	`Currency Hash` - The address of of the token (Basically the ID of the _Smart Contract_ which represents it).
	-	`Currency Type` - The type of currency (Usually `ERC20`).
	-	`Amount [USD]` - The value of the `Amount` field in USD at the time of the transaction (`Transaction Time`).
	-	`Gas Price` - The gas price allocated by the transaction initiator.
	-	`Gas Limit` - The limit on the fees (`Gas Price * Gas Consumed`) allocated by the transaction initiator.
	-	`Gas Used` - The actual amount of gas used. The actual fee paid is `Gas Used * Gas Price` and it must be below `Gas Limit` (Otherwise the VM would stop running it).
	-	`Gas Predicted` - The predicted amount of gas (Supplied by 3rd).
	
 -	Labels
	- `Label` - Sets the validity of  the transaction.  
		-	Valid Transaction - Value of `0`.
		- 	Suspicious Transaction - Value of `1`.
		-	Undetermined Transaction - Value of `-1`.
	- `Risk Value` - Sets the risk of the transaction.

 -	Smart Contracts  
	-	Assumptions:
		-	Smart Contract will have at least 2 actions with the same _Transaction Hash_.
		-	No other kind of transaction can have more than 1 action per _Transaction Hash_.
	-	Each action will have its own row in the CSV file.
	-	The feature `SmartContract` can be calculated by: `Smart Contract = num(Transaction ID) > 1` <-- Needs verification.

## Attack Types

 - 	Single Asset Single Attack (SASA).
 -	Single Asset Multiple Attacks (SAMA).
 -	Multiple Assets Multiple Attacks (MAMA).

> **Note**  
> This requires in depth discussion with the team.

## To Do

 -  [x] Standardize the Excel files into a template - **Hakan**.
 -  [x] Create the CSV files from the Excel files - **Royi**.
 -	[x]	Add the `Amount [USD]` column to each CSV - **Hakan**.
 -	[x] Add the features made by Hakan to the DF by code - **Royi**.
 -	[x] Think of the transaction time granularity - **Royi**, **Meir**.  
		Do we need granularity beyond seconds?  
		At the moment we'll use granularity of 1 [Sec]. It will also be the minimum value for activation length.
 -	[x]	Template for _Smart Contract_ - **Royi**, **Meir**.  
		Does a _Smart Contract_ require more fields? Are all fields relevant to a _Smart Contract_?  
		Royi: An idea could be adding an `Initializer ID` which is the ID of the one initialized the events and then add the events per row.
 -	[ ]
