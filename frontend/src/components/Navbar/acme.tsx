
import Icon from "./icon.png"; // import your image

export const AcmeIcon = ({ size = 24 }: { size?: number }) => (
  <img
    src={Icon}
    alt="ACME Logo"
    width={size}
    height={size}
    style={{ display: "inline-block" }}
  />
);
