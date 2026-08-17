import sys
import pandas as pd


def merge_continuous_csv(file1_path: str, file2_path: str, output_path: str):
    # Read both CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Get the last timestamp value from the second column (index 1) of the first file
    time_col_name = df1.columns[1]
    last_timestamp = df1[time_col_name].iloc[-1]

    # Offset the second column of the second dataframe to make it continuous
    df2[time_col_name] = df2[time_col_name] + last_timestamp

    # Append df2 to df1 (df2's header is naturally omitted during concatenation)
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Round the timestamp column to 1 decimal place
    merged_df[time_col_name] = merged_df[time_col_name].round(1)

    # Save to a new CSV file without writing pandas index numbers
    merged_df.to_csv(output_path, index=False)
    print(
        f"Successfully merged files. Time offset applied: {last_timestamp}. Saved to '{output_path}'."
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <path_to_csv1> <path_to_csv2> [output_csv]")
        sys.exit(1)

    csv1 = sys.argv[1]
    csv2 = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "merged_output.csv"

    merge_continuous_csv(csv1, csv2, out)